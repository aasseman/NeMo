# Copyright (C) IBM Corporation 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from math import nan
from functools import reduce

import torch
from torch import nn

from ..modules.data_layers.cog import constants


@dataclass
class TaskStats:
    n_correct: int = 0
    n_total: int = 0

    def __add__(self, other):
        return TaskStats(
            n_correct=self.n_correct + other.n_correct,
            n_total=self.n_total + other.n_total,
        )


def calculate_stats(
    tasks,
    mask_words=None,
    mask_pointings=None,
    prediction_answers=None,
    prediction_pointings=None,
    target_answers=None,
    target_pointings=None,
) -> dict:
    d = dict()

    # Classification stats (word answer)
    # If none of these are `None`
    if not any(map(lambda x: x is None, [mask_words, prediction_answers, target_answers])):
        # Get sizes.
        batch_size = prediction_answers.size(0)
        img_seq_len = prediction_answers.size(1)

        # Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
        prediction_answers = prediction_answers.view(batch_size * img_seq_len, -1)

        # Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
        target_answers = target_answers.view(batch_size * img_seq_len)

        # Retrieve "answer" and "pointing" masks, both of size [BATCH_SIZE * IMG_SEQ_LEN].
        mask_words = mask_words.view(batch_size * img_seq_len)

        _, indices = torch.max(prediction_answers, 1)

        # Calculate correct answers with additional "masking".
        correct_answers = (indices == target_answers) * mask_words

        # Iterate over each element in batch
        for (task_, correct_answers_, mask_words_) in zip(tasks, correct_answers, mask_words):
            if task_ in constants.CLASSIFICATION_TASKS:
                if task_ in d.keys():
                    d[task_].n_total += mask_words_.sum().cpu().item()
                    d[task_].n_correct += correct_answers_.sum().cpu().item()
                else:
                    d[task_] = TaskStats(
                        n_total=mask_words_.sum().cpu().item(), n_correct=correct_answers_.sum().cpu().item()
                    )

    # Regression stats (pointing)
    # If none of these are `None`
    if not any(map(lambda x: x is None, [mask_pointings, prediction_pointings, target_pointings])):
        # Get sizes.
        batch_size = prediction_pointings.size(0)
        img_seq_len = prediction_pointings.size(1)

        # Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
        prediction_pointings = prediction_pointings.view(batch_size * img_seq_len, -1)

        # Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
        target_pointings = target_pointings.view(batch_size * img_seq_len, -1)

        # Normalize pointing with softmax.
        softmax_pointing = nn.Softmax(dim=1)
        prediction_pointings = softmax_pointing(prediction_pointings)

        # Calculate mean square error for every pointing action.
        diff_pointing = target_pointings - prediction_pointings
        diff_pointing = diff_pointing ** 2
        # Sum all differences for a given answer.
        # As a results we got 1D tensor of size [BATCH_SIZE * IMG_SEQ_LEN].
        diff_pointing = torch.sum(diff_pointing, dim=1)

        # Apply  threshold.
        threshold = 0.15 ** 2

        # Check correct pointings.
        correct_pointing = (diff_pointing < threshold).type(torch.ByteTensor) * mask_pointings.flatten()
        correct_pointing = correct_pointing.view(batch_size, img_seq_len, -1)

        # Iterate over each element in batch
        for (task_, correct_pointing_, mask_pointings_) in zip(tasks, correct_pointing, mask_pointings):
            if task_ in constants.REGRESSION_TASKS:
                if task_ in d.keys():
                    d[task_].n_total += mask_pointings_.sum().cpu().item()
                    d[task_].n_correct += correct_pointing_.sum().cpu().item()
                else:
                    d[task_] = TaskStats(
                        n_total=mask_pointings_.sum().cpu().item(), n_correct=correct_pointing_.sum().cpu().item()
                    )

    return d


def collate_stats(stats: [dict]) -> dict:
    # merge keys
    keys = set()
    for d in stats:
        keys.update(d.keys())

    stats_collated = {
        k: [d[k] for d in stats if k in d.keys()] for k in keys  # list of values under key from each dict
    }

    return stats_collated


def calculate_accuracies(d: dict) -> dict:
    accuracies = dict()
    # Compute mean for each of those lists

    n_total_alltasks = 0
    n_correct_alltasks = 0

    s_class = TaskStats()
    s_reg = TaskStats()

    for k, l in d.items():
        s = reduce(lambda a, b: a + b, l)
        if k in constants.CLASSIFICATION_TASKS:
            s_class += s
        elif k in constants.REGRESSION_TASKS:
            s_reg += s

        accuracies[k] = dict()
        accuracies[k] = s.n_correct / s.n_total if s.n_total != 0 else nan

    accuracies["AllClass"] = s_class.n_correct / s_class.n_total if s_class.n_total != 0 else nan
    accuracies["AllReg"] = s_reg.n_correct / s_reg.n_total if s_reg.n_total != 0 else nan

    s = s_class + s_reg
    accuracies["All"] = s.n_correct / s.n_total if s.n_total != 0 else nan

    return accuracies
