import inspect
import os
import re
import json
from typing import Callable, Dict, List, Optional, Tuple, Union

import datasets
from sklearn.metrics import jaccard_score
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedModel, Seq2SeqTrainer, Trainer, __version__
from transformers.adapters.composition import AdapterCompositionBlock, Fuse
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available
from transformers.modeling_utils import unwrap_model

from ..configuration_utils import PretrainedConfig
from ..data.data_collator import DataCollator
from ..file_utils import CONFIG_NAME, WEIGHTS_NAME, is_sagemaker_mp_enabled, logger
from ..optimization import Adafactor, AdamW
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..trainer_callback import TrainerCallback, TrainerControl, TrainerState
from ..trainer_pt_utils import get_parameter_names
from ..trainer_utils import EvalPrediction, ShardedDDPOption
from ..training_args import TrainingArguments
from ..trainer import *


if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


class AdapterTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        adapter_names: Optional[List[List[str]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=[AdapterTrainerCallback(self)] + callbacks if callbacks else [AdapterTrainerCallback(self)],
            optimizers=optimizers,
        )

        if adapter_names is not None:
            self.model.set_active_adapters(adapter_names)
        # Set the defaults for loading/ saving model & adapters
        if isinstance(self.model, PreTrainedModel):
            model_freezed = getattr(self.model.base_model, "model_freezed", False)
        else:
            model_freezed = False
        if model_freezed and self.model.active_adapters:
            # Check if training AdapterFusion
            self.train_adapter_fusion = (
                isinstance(self.model.active_adapters, Fuse)
                or isinstance(self.model.active_adapters, AdapterCompositionBlock)
                and any([isinstance(child, Fuse) for child in self.model.active_adapters.children])
            )
        if model.active_adapters is None:
            raise ValueError(
                "Expected a model with an active adapter setup."
                "If you want to fully finetune the model use the Trainer class."
            )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if hasattr(self.model, "config") and hasattr(self.model.config, "adapters"):
                match_str = r"adapter_fusion_layer\..*\.value"
                decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_all_adapters(output_dir)
            if self.train_adapter_fusion:
                self.model.save_all_adapter_fusions(output_dir)
            if hasattr(self.model, "heads"):
                self.model.save_all_heads(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load(self, resume_from_checkpoint):
        args = self.args
        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warn(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        else:
            if os.path.isdir(resume_from_checkpoint):
                adapter_loaded = self._load_adapters(resume_from_checkpoint)
                self._load_adapter_fusions(resume_from_checkpoint)
                # Save all heads for a model with heads
                if hasattr(self.model, "heads"):
                    self._load_heads(resume_from_checkpoint)

            if not adapter_loaded:
                raise Exception("Can't find a valid checkpoint at {}".format(resume_from_checkpoint))

    def _load_adapters(self, resume_from_checkpoint):
        adapter_loaded = False
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "adapter_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_adapter(os.path.join(os.path.join(resume_from_checkpoint, file_name)))
                    adapter_loaded = True
        return adapter_loaded

    def _load_adapter_fusions(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," in file_name:
                    self.model.load_adapter_fusion(os.path.join(resume_from_checkpoint, file_name))

    def _load_heads(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "head_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_head(os.path.join(resume_from_checkpoint, file_name))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
            self._signature_columns += self.label_names
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)


class AdapterDiffTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        heldout_eval_dataset: Optional[Dataset] = None,
        diff_tasks: Optional[str] = None,
        adapter_fusion: Optional[bool] = None,
        hwu_loss: Optional[bool] = None,
        adapter_cache_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        adapter_names: Optional[List[List[str]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=[AdapterTrainerCallback(self)] + callbacks if callbacks else [AdapterTrainerCallback(self)],
            optimizers=optimizers,
        )

        self.train_dataset = train_dataset
        self.heldout_eval_dataset = heldout_eval_dataset
        self.adapter_cache_path = adapter_cache_path
        self.hwu_loss = hwu_loss

        if adapter_names is not None:
            self.model.set_active_adapters(adapter_names)
        # Set the defaults for loading/ saving model & adapters
        if isinstance(self.model, PreTrainedModel):
            model_freezed = getattr(self.model.base_model, "model_freezed", False)
        else:
            model_freezed = False
        if model_freezed and self.model.active_adapters:
            # Check if training AdapterFusion
            self.train_adapter_fusion = (
                isinstance(self.model.active_adapters, Fuse)
                or isinstance(self.model.active_adapters, AdapterCompositionBlock)
                and any([isinstance(child, Fuse) for child in self.model.active_adapters.children])
            )

            self.train_adapter_fusion = (self.train_adapter_fusion or adapter_fusion)
        if model.active_adapters is None:
            raise ValueError(
                "Expected a model with an active adapter setup."
                "If you want to fully finetune the model use the Trainer class."
            )
        
        self.layer_num = len(model.encoder.block) + len(model.decoder.block)
        self.laryerwise_candidate_adapter = dict()
        self.current_active_adapters = dict()
        task_names = diff_tasks.split('-')
        for i in range(self.layer_num):
            self.laryerwise_candidate_adapter[f'L{str(i)}'] = \
                dict([(name, f'{diff_tasks}-L{str(i)}') for name in task_names])
            self.current_active_adapters[f'L{str(i)}'] = f'{diff_tasks}-L{str(i)}'
        self.diff_task_names = task_names
        self.diff_operation = True

        # train adapter fusion with adapter differentiation
        if self.train_adapter_fusion:
            fusion_active = False
            self.laryerwise_fusion_adapters = dict()
            for i in range(self.layer_num):
                self.laryerwise_fusion_adapters[f'L{str(i)}'] = [fusion_active, f'Fusion-L{str(i)}']
        
        print('#'*50)
        print('>>> train_adapter_fusion', self.train_adapter_fusion)
                
        # Weighted sum learning
        if self.hwu_loss:
            task_num = len(task_names)
            self.huw_log_vars = [torch.zeros((1,)) for _ in range(task_num)]
            if torch.cuda.is_available():
                self.huw_log_vars = [p.cuda() for p in self.huw_log_vars]
                for p in self.huw_log_vars:
                    p.requires_grad = True
            self.huw_param_optim = torch.optim.Adam(self.huw_log_vars)
            self.huw_pairs = dict(zip(task_names, self.huw_log_vars))

    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            # labels = inputs.pop("labels")
            labels = inputs["labels"]
        else:
            labels = None
        outputs = model(**inputs)

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.hwu_loss:
            log_var = self.huw_pairs[self.current_task]
            precision = torch.exp(-log_var)
            huw_loss = torch.mean(precision * loss + log_var)
            loss = huw_loss

        return (loss, outputs) if return_outputs else loss

    def update_optimizer_params_groups(self):
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if hasattr(self.model, "config") and hasattr(self.model.config, "adapters"):
            match_str = r"adapter_fusion_layer\..*\.value"
            decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]

        add_decay_params = [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad and n not in self.candidate_params]
        add_not_decay_params = [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad and n not in self.candidate_params]

        # add_decay_params = [p for n, p in self.model.named_parameters() if n in decay_parameters]
        # add_not_decay_params = [p for n, p in self.model.named_parameters() if n not in decay_parameters]

        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.candidate_params:
                self.candidate_params.append(n)

        optimizer_grouped_parameters = [
            {
                "params": add_decay_params,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": add_not_decay_params,
                "weight_decay": 0.0,
            },
        ]

        # self.optimizer.param_groups.clear()
        # self.optimizer.state.clear()

        for param_group in optimizer_grouped_parameters:
            self.optimizer.add_param_group(param_group)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if hasattr(self.model, "config") and hasattr(self.model.config, "adapters"):
                match_str = r"adapter_fusion_layer\..*\.value"
                decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]

            self.candidate_params = [n for n, p in self.model.named_parameters()]
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)

    def _deactivate_adapter_runtime(self, adapter_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model.save_adapter(save_path, adapter_name)
        self.model.delete_adapter(adapter_name)

    def _deactivate_adapter_fusion_runtime(self, adapter_fusion_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_fusion_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model.save_adapter_fusion(save_path, adapter_fusion_name)
        self.model.delete_adapter_fusion(adapter_fusion_name)

    def _activate_adapter_runtime(self, adapter_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model.load_adapter(save_path, load_as=adapter_name, set_active=True)

    def _copy_adapter_fusion_runtime(self, new_adapter_fusion_name, adapter_fusion_name):
        save_path = os.path.join(self.adapter_cache_path, adapter_fusion_name)
        assert os.path.exists(save_path)
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # self.model.save_adapter_fusion(save_path, adapter_fusion_name)
        self.model.load_adapter_fusion(save_path, load_as=new_adapter_fusion_name, set_active=True)

    def _create_adapter_fusion_runtime(self, adapter_fusion_name):
        self.model.add_adapter_fusion(adapter_fusion_name, set_active=True)

    def _copy_adapter_runtime(self, target_adapter, source_adapter):
        save_path = os.path.join(self.adapter_cache_path, source_adapter)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model.load_adapter(save_path, load_as=target_adapter, set_active=True)
        
        save_path = os.path.join(self.adapter_cache_path, target_adapter)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.model.save_adapter(save_path, target_adapter)

    def _extract_adapter_grads(self, task_name):
        # record all the adapter gradients on held-out evaluation data
        # print('#'*100)
        # print('>>> TARGET TASK',task_name)
        for name, param in self.model.named_parameters():
            # if 'fusion' in name and param.requires_grad:
            #     print('@@@FUSION', name)
            #     if param.grad is not None:
            #         print('### HERE ###')
            #         exit(0)
                    
            if 'adapter' in name and (f'.{task_name}-' in name or f'-{task_name}-' in name) and ',' not in name and param.requires_grad:
                layer = name.split('.')[6].split('-')[-1]
                if layer not in self.exist_adapter_cell_grads:
                    self.exist_adapter_cell_grads[layer] = {}
                if task_name not in self.exist_adapter_cell_grads[layer]:
                    self.exist_adapter_cell_grads[layer][task_name] = []

                self.exist_adapter_cell_grads[layer][task_name].append(param.grad.clone().detach().view(1, -1))

    def _switch_model_task_mode(self, target_task):
        # Switch the model on the target task mode
        self.current_task = target_task

        if not self.train_adapter_fusion:
            adapter_names = []
            for layer, adapter_name in self.current_active_adapters.items():
                if adapter_name.startswith(f'{target_task}-') or f'-{target_task}-' in adapter_name:
                    adapter_names.append(adapter_name)
                    continue
                else:
                    if len(adapter_name) > 0:
                        self._deactivate_adapter_runtime(adapter_name)
                    target_adapter = self.laryerwise_candidate_adapter[layer][target_task]
                    adapter_names.append(target_adapter)
                    self._activate_adapter_runtime(target_adapter)
                    self.current_active_adapters[layer] = target_adapter
            
            # print('Switch to', target_task, adapter_names)
            self.model.train_adapter(adapter_names)
        else:
            adapter_fusion_names = []
            undiff_adapters = []
            for layer, fusion_state in self.laryerwise_fusion_adapters.items():
                if fusion_state[0]:
                    adapter_fusion_names.append(fusion_state[1].split(','))
                else:
                    undiff_adapter = self.laryerwise_candidate_adapter[layer][self.current_task]
                    undiff_adapters.append(undiff_adapter)

            if len(adapter_fusion_names) > 0:
                self.model.train_adapter_and_fusion(undiff_adapters, adapter_fusion_names, unfreeze_adapters=True, target_task=target_task)
            else:
                self.model.train_adapter(undiff_adapters)

        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))

        self.update_optimizer_params_groups()            

    def _find_differentiatable_cell(self):
        # find all the differentiatable cells according to the
        #  record 'self.exist_adapter_cell_grads'
        # MAIN algorithem in the paper
        # output: {'original_cell': List(differentiated cells)}
        # update global active cells
        def _calculate_interference_degree(task_grad_mapping):
            shared_task_len = len(task_grad_mapping)

            assert shared_task_len > 1

            task_grads = torch.stack([g.view(-1,) for g in task_grad_mapping.values()])

            # print('TAG', task_grads)

            dot_prod = torch.mm(task_grads, task_grads.t())
            norm = torch.norm(task_grads, p=2, dim=1).unsqueeze(0)
            cos = dot_prod.div(torch.mm(norm.t(), norm))
            interference_degree = []
            for i in range(shared_task_len-1):
                # interference degree equals to nagetive cosine similarity
                interference_degree.append(-cos[i, i+1:])
            interference_degree = torch.cat(interference_degree)

            return list(task_grad_mapping.keys()), interference_degree

        differentiated_cell_mapping = {}
        for layer, task_grads in self.exist_adapter_cell_grads.items():
            laryerwise_adapter_grad_mapping = {}
            for task_name, task_grad in task_grads.items():
                adapter_name = self.laryerwise_candidate_adapter[layer][task_name]
                if adapter_name not in laryerwise_adapter_grad_mapping:
                    laryerwise_adapter_grad_mapping[adapter_name] = {}

                task_grad = torch.cat(task_grad, dim=1)
                laryerwise_adapter_grad_mapping[adapter_name][task_name] = task_grad

            for adapter_name, task_grad_mapping in laryerwise_adapter_grad_mapping.items():
                if len(task_grad_mapping) == 1:
                    continue
                else:
                    tasks, interference_degrees = _calculate_interference_degree(task_grad_mapping)
                    # print('@@@', interference_degrees)
                    max_interference_degree = torch.max(interference_degrees)
                    task_len = len(tasks)
                    if max_interference_degree > self.args.max_interference_degree:
                        if layer not in differentiated_cell_mapping:
                            differentiated_cell_mapping[adapter_name] = {}
                        # start to differentiate
                        flag = 0
                        group1 = []
                        group2 = []
                        task_distance = {}
                        for i in range(task_len-1):
                            for j in range(i+1, task_len):
                                if interference_degrees[flag] == max_interference_degree:
                                    group1.append(i)
                                    group2.append(j)
                                
                                task_distance[(i,j)] = interference_degrees[flag]
                                flag += 1

                        for i in range(task_len):
                            if i in group1 or i in group2:
                                continue
                            distance_to_g1 = []
                            for j in group1:
                                a = i
                                b = j
                                if i == j:
                                    continue
                                if i > j:
                                    a,b = b,a
                                distance_to_g1.append(task_distance[(a,b)])

                            distance_to_g2 = []
                            for k in group2:
                                a = i
                                b = k
                                if i == k:
                                    continue
                                if i > k:
                                    a,b = b,a
                                distance_to_g2.append(task_distance[(a,b)])
                            
                            distance_to_g1 = torch.stack(distance_to_g1).view(-1,)
                            distance_to_g2 = torch.stack(distance_to_g2).view(-1,)

                            if torch.max(distance_to_g1) < torch.max(distance_to_g2):
                                group1.append(i)
                            else:
                                group2.append(i)

                        group1 = [tasks[t] for t in group1]
                        group2 = [tasks[t] for t in group2]

                        differentiated_cell_mapping[adapter_name] = [group1, group2]
        return differentiated_cell_mapping

    def _update_differentiated_model(self, differentiated_cells):
        # add new differentiated cells in the model and load 
        # the corresponding params in the optimizer
        for adapter_name, split_group in differentiated_cells.items():
            layer = adapter_name.split('-')[-1]
            adapter_group1 = '-'.join(split_group[0]) + f'-{layer}'
            adapter_group2 = '-'.join(split_group[1]) + f'-{layer}'

            if adapter_name == self.current_active_adapters[layer]:
                self._deactivate_adapter_runtime(adapter_name)
            self._copy_adapter_runtime(adapter_group1, adapter_name)
            self._copy_adapter_runtime(adapter_group2, adapter_name)

            self.current_active_adapters[layer] = ''

            for task in split_group[0]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group1

            for task in split_group[1]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group2

    def _update_differentiated_fusion_model(self, differentiated_cells):
        # add new differentiated cells in the model and load 
        # the corresponding params in the optimizer

        processed_fusion_layer = []
        for adapter_name, split_group in differentiated_cells.items():
            layer = adapter_name.split('-')[-1]
            adapter_group1 = '-'.join(split_group[0]) + f'-{layer}'
            adapter_group2 = '-'.join(split_group[1]) + f'-{layer}'

            layer_fusion_active = self.laryerwise_fusion_adapters[layer][0]

            self._deactivate_adapter_runtime(adapter_name)
            if layer_fusion_active and layer not in processed_fusion_layer:
                self._deactivate_adapter_fusion_runtime(self.laryerwise_fusion_adapters[layer][1])
                processed_fusion_layer.append(layer)
            
            self._copy_adapter_runtime(adapter_group1, adapter_name)
            self._copy_adapter_runtime(adapter_group2, adapter_name)

            for task in split_group[0]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group1

            for task in split_group[1]:
                self.laryerwise_candidate_adapter[layer][task] = adapter_group2
        
        for adapter_name in differentiated_cells.keys():
            layer = adapter_name.split('-')[-1]
            layer_fusion_active = self.laryerwise_fusion_adapters[layer][0]

            layer_adapters = list(set(list(self.laryerwise_candidate_adapter[layer].values())))
            layer_fusion_name = ','.join(layer_adapters)
            if not layer_fusion_active:
                self._create_adapter_fusion_runtime(layer_fusion_name)
                self.laryerwise_fusion_adapters[layer][0] = True
            else:
                self._copy_adapter_fusion_runtime(layer_fusion_name, self.laryerwise_fusion_adapters[layer][1])
            
            self.laryerwise_fusion_adapters[layer][1] = layer_fusion_name
        
    def _differentiate_operate(self):
        self.exist_adapter_cell_grads = {}

        self.heldout_dataloader = self.get_eval_dataloader(self.heldout_eval_dataset)

        for current_task in self.diff_task_names:
            self._switch_model_task_mode(current_task)
            self.optimizer.zero_grad()
            for heldout_data in self.heldout_dataloader:
                task_name = heldout_data['tasks'][0]
                if task_name != current_task:
                    continue

                heldout_data = {
                    'input_ids': heldout_data['input_ids'].cuda(),
                    'attention_mask': heldout_data['attention_mask'].cuda(),
                    'labels': heldout_data['labels'].cuda(),
                }
                # Calculate the gradient of the given heldout evaluation data
                loss = self.compute_loss(self.model, heldout_data)
                loss.backward()
            
            # print('>>> LOSS', loss)
            self._extract_adapter_grads(current_task)
        
        diff_cells = self._find_differentiatable_cell()
        # print(diff_cells)

        if not self.train_adapter_fusion:
            self._update_differentiated_model(diff_cells)
        else:
            self._update_differentiated_fusion_model(diff_cells)

    def _calculate_differentiated_rate(self):
        initial_adapter_num = len(self.laryerwise_candidate_adapter)
        current_adapter_names = []
        for layer in self.laryerwise_candidate_adapter.keys():
            for task_name in self.laryerwise_candidate_adapter[layer].keys():
                current_adapter_names.append(self.laryerwise_candidate_adapter[layer][task_name])
        
        current_adapter_num = len(list(set(current_adapter_names)))

        return current_adapter_num / initial_adapter_num

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            self._load(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        # train_dataloader = self.get_train_dataloader()
        train_dataloader = self.get_eval_dataloader(self.train_dataset)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            self.current_task = self.diff_task_names[0]
            differentiate_detect_step = args.differentiate_detect_step
            differentiate_start_step = args.differentiate_start_step

            for step, inputs in enumerate(epoch_iterator):
                if inputs['tasks'][0] != self.current_task:
                    # print('>>>> Switch at step', step)
                    # self.current_task = inputs['tasks'][0]
                    self._switch_model_task_mode(inputs['tasks'][0])
                
                inputs = {
                    'input_ids': inputs['input_ids'].cuda(),
                    'attention_mask': inputs['attention_mask'].cuda(),
                    'labels': inputs['labels'].cuda(),
                }

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        if self.hwu_loss:
                            self.huw_param_optim.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    if self.state.global_step > differentiate_start_step and (step + 1) % differentiate_detect_step == 0 and self.diff_operation:
                        self._differentiate_operate()
                        current_diff_rate = self._calculate_differentiated_rate()
                        if current_diff_rate >= self.args.differentiate_rate_threshold:
                             self.diff_operation = False
                        self._switch_model_task_mode(self.current_task)

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # if not isinstance(self.model, PreTrainedModel):
        #     if isinstance(unwrap_model(self.model), PreTrainedModel):
        #         if state_dict is None:
        #             state_dict = self.model.state_dict()
        #         unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
        #     else:
        #         logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        #         if state_dict is None:
        #             state_dict = self.model.state_dict()
        #         torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        # else:
        #     for task_name in self.diff_task_names:
        #         self._switch_model_task_mode(task_name)
        #         self.model.save_all_adapters(output_dir)
        #     if self.train_adapter_fusion:
        #         self.model.save_all_adapter_fusions(output_dir)
        #     if hasattr(self.model, "heads"):
        #         self.model.save_all_heads(output_dir)

        if not self.train_adapter_fusion:
            activate_adapters = []
            for layer in self.laryerwise_candidate_adapter.keys():
                for target_task in self.laryerwise_candidate_adapter[layer].keys():
                    activate_adapters.append(self.laryerwise_candidate_adapter[layer][target_task])
            
            activate_adapters = list(set(activate_adapters))

            current_activate_adapters = list(self.current_active_adapters.values())

            for adapter in activate_adapters:
                if adapter in current_activate_adapters:
                    self.model.save_adapter(os.path.join(output_dir, adapter), adapter)
                else:
                    adapter_path = f'{self.adapter_cache_path}/{adapter}'
                    os.system(f'cp -rf {adapter_path} {output_dir}')
        else:
            activate_adapter_fusions = []
            for layer in self.laryerwise_candidate_adapter.keys():
                layer_activate_adapters = []
                for target_task in self.laryerwise_candidate_adapter[layer].keys():
                    layer_activate_adapters.append(self.laryerwise_candidate_adapter[layer][target_task])
                if len(set(layer_activate_adapters)) == 1:
                    adapter = layer_activate_adapters[0]
                    self.model.save_adapter(os.path.join(output_dir, adapter), adapter)
                else:
                    adapter_fusion = self.laryerwise_fusion_adapters[layer][1]
                    activate_adapter_fusions.append(adapter_fusion)
                    self.model.save_adapter_fusion(os.path.join(output_dir, adapter_fusion), adapter_fusion)
                    for adapter in adapter_fusion.split(','):
                        self.model.save_adapter(os.path.join(output_dir, adapter), adapter)
            json.dump(activate_adapter_fusions, open(os.path.join(output_dir, 'activate_adapter_fusions.json'), 'w'), indent=4)
            
        json.dump(self.laryerwise_candidate_adapter, open(os.path.join(output_dir, 'adapter_structure.json'), 'w'), indent=4)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load(self, resume_from_checkpoint):
        args = self.args
        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warn(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        else:
            if os.path.isdir(resume_from_checkpoint):
                adapter_loaded = self._load_adapters(resume_from_checkpoint)
                self._load_adapter_fusions(resume_from_checkpoint)
                # Save all heads for a model with heads
                if hasattr(self.model, "heads"):
                    self._load_heads(resume_from_checkpoint)

            if not adapter_loaded:
                raise Exception("Can't find a valid checkpoint at {}".format(resume_from_checkpoint))

    def _load_adapters(self, resume_from_checkpoint):
        adapter_loaded = False
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "adapter_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_adapter(os.path.join(os.path.join(resume_from_checkpoint, file_name)))
                    adapter_loaded = True
        return adapter_loaded

    def _load_adapter_fusions(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," in file_name:
                    self.model.load_adapter_fusion(os.path.join(resume_from_checkpoint, file_name))

    def _load_heads(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "head_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_head(os.path.join(resume_from_checkpoint, file_name))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
            self._signature_columns += self.label_names
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)


class AdapterTrainerCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.pop("model")
        model_freezed = getattr(model.base_model, "model_freezed", False)
        if not model_freezed:
            raise ValueError("The model is not freezed. For training adapters please call the train_adapters() method")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.pop("model")
        if args.load_best_model_at_end and state.best_model_checkpoint is not None:

            logger.info(f"Loading best adapter(s) from {state.best_model_checkpoint} (score: {state.best_metric}).")
            # attempt to re-load all adapters from checkpoint
            for adapter in model.config.adapters.adapters:
                adapter_dir = os.path.join(state.best_model_checkpoint, adapter)
                if os.path.exists(adapter_dir):
                    model.load_adapter(adapter_dir)
            if self.trainer.train_adapter_fusion:
                logger.info(
                    f"Loading best adapter fusion(s) from {state.best_model_checkpoint} (score: {state.best_metric})."
                )
                # attempt to re-load all adapter fusions from checkpoint
                fusion_models = getattr(self.model.config, "adapter_fusion_models", [])
                for fusion in fusion_models:
                    fusion_dir = os.path.join(state.best_model_checkpoint, fusion)
                    if os.path.exists(fusion_dir):
                        self.model.load_adapter_fusion(fusion_dir)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # apply adapter fusion weight regularization on the value matrix
        model = kwargs.pop("model")
        # if self.trainer.train_adapter_fusion:
        #     fusion_reg_loss = model.base_model.get_fusion_regularization_loss()
        #     fusion_reg_loss.backward()


class Seq2SeqAdapterTrainer(AdapterTrainer, Seq2SeqTrainer):
    pass

class Seq2SeqAdapterDiffTrainer(AdapterDiffTrainer, Seq2SeqTrainer):
    pass
