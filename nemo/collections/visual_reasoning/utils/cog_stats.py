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
    n_correct_class: int = 0
    n_total_class: int = 0
    n_correct_reg: float = 0
    n_total_reg: float = 0

    def __add__(self, other):
        return TaskStats(
            n_correct_class=self.n_correct_class + other.n_correct_class,
            n_total_class=self.n_total_class + other.n_total_class,
            n_correct_reg=self.n_correct_reg + other.n_correct_reg,
            n_total_reg=self.n_total_reg + other.n_total_reg
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
            if task_ in d.keys():
                d[task_].n_total_class += mask_words_.sum().cpu().item()
                d[task_].n_correct_class += correct_answers_.sum().cpu().item()
            else:
                d[task_] = TaskStats(
                    n_total_class=mask_words_.sum().cpu().item(), n_correct_class=correct_answers_.sum().cpu().item()
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
            if task_ in d.keys():
                d[task_].n_total_reg += mask_pointings_.sum().cpu().item()
                d[task_].n_correct_reg += correct_pointing_.sum().cpu().item()
            else:
                d[task_] = TaskStats(
                    n_total_reg=mask_pointings_.sum().cpu().item(), n_correct_reg=correct_pointing_.sum().cpu().item()
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

    n_total_class_alltasks = 0
    n_correct_class_alltasks = 0
    n_total_reg_alltasks = 0
    n_correct_reg_alltasks = 0

    s_alltasks = TaskStats()

    for k, l in d.items():
        s = reduce(lambda a, b: a + b, l)
        s_alltasks += s

        accuracies[k] = dict()
        accuracies[k]["class"] = s.n_correct_class / s.n_total_class if s.n_total_class != 0 else nan
        accuracies[k]["reg"] = s.n_correct_reg / s.n_total_reg if s.n_total_reg != 0 else nan
        accuracies[k]["all"] = (
            (s.n_correct_class + s.n_correct_reg) / (s.n_total_class + s.n_total_reg)
            if (s.n_total_class + s.n_total_reg) != 0
            else nan
        )

    accuracies["AllTasks"] = dict()
    accuracies["AllTasks"]["class"] = (
        s_alltasks.n_correct_class / s_alltasks.n_total_class if s_alltasks.n_total_class != 0 else nan
    )
    accuracies["AllTasks"]["reg"] = s_alltasks.n_correct_reg / s_alltasks.n_total_reg if s_alltasks.n_total_reg != 0 else nan
    accuracies["AllTasks"]["all"] = (
        (s_alltasks.n_correct_class + s_alltasks.n_correct_reg) / (s_alltasks.n_total_class + s_alltasks.n_total_reg)
        if (s_alltasks.n_total_class + s_alltasks.n_total_reg) != 0
        else nan
    )

    return accuracies
