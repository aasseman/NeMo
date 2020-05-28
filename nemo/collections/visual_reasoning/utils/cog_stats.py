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

import torch
from torch import nn

from ..modules.data_layers.cog import constants

# TODO: Test those functions


def calculate_accuracy(
    mask_words,
    mask_pointings,
    prediction_answers,
    prediction_pointings,
    target_answers,
    target_pointings,
    mask_pointers,
):
    """ Calculates accuracy equal to mean number of correct predictions in a given batch.
    WARNING: Applies mask to both logits and targets!
    """
    # Get sizes.
    batch_size = prediction_answers.size(0)
    img_seq_len = prediction_answers.size(1)

    # Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
    prediction_answers = prediction_answers.view(batch_size * img_seq_len, -1)
    prediction_pointings = prediction_pointings.view(batch_size * img_seq_len, -1)

    # Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
    target_answers = target_answers.view(batch_size * img_seq_len)
    # Reshape targets: pointings [BATCH_SIZE * IMG_SEQ_LEN x NUM_ACTIONS]
    target_pointings = target_pointings.view(batch_size * img_seq_len, -1)

    # Retrieve "answer" and "pointing" masks, both of size [BATCH_SIZE * IMG_SEQ_LEN].
    mask_words = mask_words.view(batch_size * img_seq_len)

    mask_pointings = mask_pointings.view(batch_size * img_seq_len)

    # print("targets_answer = ", targets_answer)
    # print("preds_answer = ", preds_answer)
    # print("mask_answer = ", mask_answer)

    # print("targets_pointing = ", targets_pointing)
    # print("preds_pointing = ", preds_pointing)
    # print("mask_pointing = ", mask_pointing)

    #########################################################################
    # Calculate accuracy for Answering task.
    # Get answers [BATCH_SIZE * IMG_SEQ_LEN]
    _, indices = torch.max(prediction_answers, 1)

    # Calculate correct answers with additional "masking".
    correct_answers = (indices == target_answers).type(torch.ByteTensor) * mask_words

    # Calculate accurary.
    if mask_words.sum() > 0:
        acc_answer = float(correct_answers.sum().item()) / float(mask_words.sum().item())
    else:
        acc_answer = 0.0

    #########################################################################
    # Calculate accuracy for Pointing task.

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
    correct_pointing = (diff_pointing < threshold).type(torch.ByteTensor) * mask_pointings
    # print('corect poitning',correct_pointing)
    # Calculate accurary.
    if mask_pointings.sum() > 0:
        acc_pointing = float(correct_pointing.sum().item()) / float(mask_pointings.sum().item())
    else:
        acc_pointing = 0.0

    #########################################################################
    # Total accuracy.
    acc_total = float(correct_answers.sum() + correct_pointing.sum()) / float(mask_words.sum() + mask_pointings.sum())
    # acc_total = torch.mean(torch.cat( (correct_answers.type(torch.FloatTensor), correct_pointing.type(torch.FloatTensor)) ) )

    # Return all three of them.
    return acc_total, acc_answer, acc_pointing


def get_acc_per_family(
    tasks,
    mask_words,
    mask_pointings,
    prediction_answers,
    prediction_pointings,
    target_answers,
    target_pointings,
    mask_pointers,
):
    """
    Compute the accuracy per family for the current batch. Also accumulates
    the number of correct predictions & questions per family in self.correct_pred_families (saved
    to file).


    .. note::

        To refactor.
    """

    # build dictionary to store acc families stats
    tuple_list = [[0, 0, 0] for _ in range(len(constants.CATEGORIES))]
    categories_stats = dict(zip(constants.CATEGORIES, tuple_list))

    # Get sizes.
    batch_size = prediction_answers.size(0)
    img_seq_len = prediction_answers.size(1)

    # Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
    prediction_answers = prediction_answers.view(batch_size * img_seq_len, -1)
    prediction_pointings = prediction_pointings.view(batch_size * img_seq_len, -1)

    # Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
    target_answers = target_answers.view(batch_size * img_seq_len)
    # Reshape targets: pointings [BATCH_SIZE * IMG_SEQ_LEN x NUM_ACTIONS]
    target_pointings = target_pointings.view(batch_size * img_seq_len, -1)

    # Retrieve "answer" and "pointing" masks, both of size [BATCH_SIZE * IMG_SEQ_LEN].
    mask_answer_non_flatten = mask_words
    mask_words = mask_words.view(batch_size * img_seq_len)

    mask_pointing_non_flatten = mask_pointings
    mask_pointings = mask_pointings.view(batch_size * img_seq_len)

    #########################################################################
    # Calculate accuracy for Answering task.
    # Get answers [BATCH_SIZE * IMG_SEQ_LEN]
    _, indices = torch.max(prediction_answers, 1)

    # Calculate correct answers with additional "masking".
    correct_answers = (indices == target_answers).type(torch.ByteTensor) * mask_words

    #########################################################################
    # Calculate accuracy for Pointing task.

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
    correct_pointing = (diff_pointing < threshold).type(torch.ByteTensor) * mask_pointings

    # count correct and total for each category
    for i in range(batch_size):

        # update # of questions for the corresponding family

        # classification
        correct_ans = correct_answers.view(batch_size, img_seq_len, -1)
        categories_stats[tasks[i]][1] += float(correct_ans[i].sum().item())

        # pointing
        correct_pointing_non_flatten = correct_pointing.view(batch_size, img_seq_len, -1)
        categories_stats[tasks[i]][1] += float(correct_pointing_non_flatten[i].sum().item())

        # update the # of correct predictions for the corresponding family

        # classification
        categories_stats[tasks[i]][0] += float(mask_answer_non_flatten[i].sum().item())

        # pointing
        categories_stats[tasks[i]][0] += float(mask_pointing_non_flatten[i].sum().item())

        # put task accuracy in third position of the dictionary
        if categories_stats[tasks[i]][0] == 0:
            categories_stats[tasks[i]][2] = 0.0

        else:
            categories_stats[tasks[i]][2] = categories_stats[tasks[i]][1] / categories_stats[tasks[i]][0]

    return categories_stats


def collect_statistics(
    tasks,
    mask_words,
    mask_pointings,
    prediction_answers,
    prediction_pointings,
    target_answers,
    target_pointings,
    mask_pointers,
    loss_answer,
    loss_pointing,
):
    stats_dict = dict()

    # Additional loss.
    loss_answer = loss_answer.cpu().item()
    loss_pointing = loss_pointing.cpu().item()

    # Accuracies.
    acc_total, acc_answer, acc_pointing = calculate_accuracy(
        mask_words,
        mask_pointings,
        prediction_answers,
        prediction_pointings,
        target_answers,
        target_pointings,
        mask_pointers,
    )
    stats_dict['acc'] = acc_total
    stats_dict['acc_answer'] = acc_answer
    stats_dict['acc_pointing'] = acc_pointing

    # Families Accuracies
    families_accuracies_dic = get_acc_per_family(
        tasks,
        mask_words,
        mask_pointings,
        prediction_answers,
        prediction_pointings,
        target_answers,
        target_pointings,
        mask_pointers,
    )

    for key in families_accuracies_dic:
        stats_dict[key] = families_accuracies_dic[key][2]

    return stats_dict
