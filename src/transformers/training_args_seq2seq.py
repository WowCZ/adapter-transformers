# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from typing import Optional

from .file_utils import add_start_docstrings
from .training_args import TrainingArguments


logger = logging.getLogger(__name__)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use a `sortish sampler` or not. Only possible if the underlying datasets are `Seq2SeqDataset` for
        now but will become generally available in the near future.

        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    generation_max_length (:obj:`int`, `optional`):
        The :obj:`max_length` to use on each evaluation loop when :obj:`predict_with_generate=True`. Will default to
        the :obj:`max_length` value of the model configuration.
    generation_num_beams (:obj:`int`, `optional`):
        The :obj:`num_beams` to use on each evaluation loop when :obj:`predict_with_generate=True`. Will default to the
        :obj:`num_beams` value of the model configuration.
    """

    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )

    differentiate_detect_step: Optional[int] = field(
        default=1000,
        metadata={
            "help": "differentiation detect step"
        },
    )

    max_interference_degree: Optional[float] = field(
        default=0,
        metadata={
            "help": "max interference degree"
        },
    )

    min_intra_simiarity: Optional[float] = field(
        default=-1,
        metadata={
            "help": "max interference degree"
        },
    )

    max_entropy_threshold: Optional[float] = field(
        default=-1,
        metadata={
            "help": "max interference degree"
        },
    )

    differentiate_start_step: Optional[int] = field(
        default=0,
        metadata={
            "help": "differentiation starting step"
        },
    )

    differentiate_rate_threshold: Optional[float] = field(
        default=5,
        metadata={
            "help": "differentiate rate threshold"
        },
    )
