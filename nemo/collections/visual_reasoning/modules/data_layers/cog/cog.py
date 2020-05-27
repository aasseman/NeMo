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


"""cog.py: Implementation of Google's COG dataset. https://arxiv.org/abs/1803.06092"""

__author__ = "Emre Sevgen, Tomasz Kornuta"

import gzip
import json
import os
import string
import tarfile
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import wget
from torch.utils.data import Dataset

from .cog_utils import constants, generate_dataset
from .cog_utils import json_to_img as jti
from nemo.utils import logging


class COG(Dataset):
    """
    The COG dataset is a sequential VQA dataset.

    Inputs are a sequence of images of simple shapes and characters on a black background, \
    and a question based on these objects that relies on memory which has to be answered at every step of the sequence.

    See https://arxiv.org/abs/1803.06092 (`A Dataset and Architecture for Visual Reasoning with a Working Memory`)\
    for the reference paper.

    """

    def __init__(
        self,
        data_folder: str = "~/data/cog",
        subset: str = "train",
        cog_tasks: str = "class",
        cog_type: str = "canonical",
        cog_gen_examples_per_task=None,
        cog_gen_sequence_length=None,
        cog_gen_memory_length=None,
        cog_gen_max_distractors=None,
        cog_gen_threads=None,
    ):
        """
        Initializes the :py:class:`COG` problem:

            - Calls :py:class:`miprometheus.problems.VQAProblem` class constructor,
            - Sets the following attributes using the provided ``params``:

                - ``self.data_folder`` (`string`) : Data directory where the dataset is stored.
                - ``self.set`` (`string`) : 'val', 'test', or 'train'
                - ``self.tasks`` (`string` or list of `string`) : Which tasks to use. 'class', 'reg', \
                'all', 'binary', or a list of tasks such as ['AndCompareColor', 'AndCompareShape']. \
                Only the selected tasks will be used.

                Classification tasks are: ['AndCompareColor', 'AndCompareShape', 'AndSimpleCompareColor',
                'AndSimpleCompareShape', 'CompareColor', 'CompareShape', 'Exist',
                'ExistColor', 'ExistColorOf', 'ExistColorSpace', 'ExistLastColorSameShape',
                'ExistLastObjectSameObject', 'ExistLastShapeSameColor', 'ExistShape',
                'ExistShapeOf', 'ExistShapeSpace', 'ExistSpace', 'GetColor', 'GetColorSpace',
                'GetShape', 'GetShapeSpace', 'SimpleCompareColor', 'SimpleCompareShape']		

                Regression tasks are: 		self.regression_tasks = ['AndSimpleExistColorGo', 'AndSimpleExistGo', 'AndSimpleExistShapeGo', 'CompareColorGo',
                'CompareShapeGo', 'ExistColorGo', 'ExistColorSpaceGo', 'ExistGo', 'ExistShapeGo',
                'ExistShapeSpaceGo', 'ExistSpaceGo', 'Go', 'GoColor', 'GoColorOf', 'GoShape',
                'GoShapeOf', 'SimpleCompareColorGo', 'SimpleCompareShapeGo', 'SimpleExistColorGo',
                'SimpleExistGo','SimpleExistShapeGo']

                Binary classification tasks are: ['AndCompareColor', 'AndCompareShape', 'AndSimpleCompareColor', 'AndSimpleCompareShape', 'CompareColor', 'CompareShape', 'Exist', 
                'ExistColor', 'ExistColorOf', 'ExistColorSpace', 'ExistLastColorSameShape', 'ExistLastObjectSameObject', 'ExistLastShapeSameColor', 
                'ExistShape', 'ExistShapeOf', 'ExistShapeSpace', 'ExistSpace', 'SimpleCompareColor', 'SimpleCompareShape'] 


                - ``self.dataset_type`` (`string`) : Which dataset to use, 'canonical', 'hard', or \
                'generated'. If 'generated', please specify 'examples_per_task', 'sequence_length', \
                'memory_length', and 'max_distractors' under 'generation'. Can also specify 'nr_processors' for generation.

            - Adds the following as default params:

                >>> {'data_folder': os.path.expanduser('~/data/cog'),
                >>>  'set': 'train',
                >>>  'tasks': 'class',
                >>>  'dataset_type': 'canonical',
                >>>  'initialization_only': False}

            - Sets:

                >>> self.data_definitions = {'images': {'size': [-1, self.sequence_length, 3, self.img_size, self.img_size], 'type': [torch.Tensor]},
                >>>                          'tasks': {'size': [-1, 1], 'type': [list, str]},
                >>>                          'questions': {'size': [-1, 1], 'type': [list, str]},
                >>>                          'targets_pointing': {'size': [-1, self.sequence_length, 2], 'type': [torch.Tensor]},
                >>>                          'targets_answer': {'size': [-1, self.sequence_length, 1], 'type' : [list,str]}
                >>>                         }



        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type params: :py:class:`miprometheus.utils.ParamInterface`


        """

        # Call base class constructors
        super(COG, self).__init__()

        # Data folder main is /path/cog
        # Data folder parent is data_X_Y_Z
        # Data folder children are train_X_Y_Z, test_X_Y_Z, or val_X_Y_Z
        self.data_folder_main = os.path.expanduser(data_folder)

        self.set = subset
        assert self.set in [
            'val',
            'test',
            'train',
        ], f"subset must be one of 'val', 'test', or 'train', got {self.set}"
        self.dataset_type = cog_type
        assert self.dataset_type in ['canonical', 'hard', 'generated'], (
            "dataset in configuration file must be one of "
            f"'canonical', 'hard', or 'generated', got {self.dataset_type}"
        )

        # Default output vocab
        self.output_vocab = constants.OUTPUTVOCABULARY

        self.tasks = cog_tasks
        if self.tasks == 'class':
            self.tasks = constants.CLASSIFICATION_TASKS
        elif self.tasks == 'reg':
            self.tasks = constants.REGRESSION_TASKS
            self.output_vocab = []
        elif self.tasks == 'all':
            self.tasks = constants.CLASSIFICATION_TASKS + constants.REGRESSION_TASKS
        elif self.tasks == 'binary':
            self.tasks = constants.BINARY_TASKS
            self.output_vocab = ['true', 'false']

        self.input_words = len(constants.INPUTVOCABULARY)
        self.output_classes = len(self.output_vocab)

        # If loading a default dataset, set default path names and set sequence length
        if self.dataset_type == 'canonical':
            self.examples_per_task = 227280
            self.sequence_length = 4
            self.memory_length = 3
            self.max_distractors = 1
        elif self.dataset_type == 'hard':
            self.examples_per_task = 227280
            self.sequence_length = 8
            self.memory_length = 7
            self.max_distractors = 10
        elif self.dataset_type == 'generated':
            self.examples_per_task = int(cog_gen_examples_per_task)
            self.sequence_length = int(cog_gen_sequence_length)
            self.memory_length = int(cog_gen_memory_length)
            self.max_distractors = int(cog_gen_max_distractors)
            self.nr_processors = int(cog_gen_threads)

        self.dataset_name = f"{self.sequence_length}_{self.memory_length}_{self.max_distractors}"
        self.data_folder_parent = os.path.join(self.data_folder_main, 'data_' + self.dataset_name)
        self.data_folder_child = os.path.join(self.data_folder_parent, self.set + '_' + self.dataset_name)

        # This should be the length of the longest sentence encounterable
        self.nwords = 24

        # Get the "hardcoded" image width/height.
        self.img_size = 112

        self.output_classes_pointing = 49

        # Set default values
        self.default_values = {
            'height': self.img_size,
            'width': self.img_size,
            'num_channels': 3,
            'sequence_length': self.sequence_length,
            'nb_classes': self.output_classes,
            'nb_classes_pointing': self.output_classes_pointing,
            'embed_vocab_size': self.input_words,
        }

        # Set data dictionary based on parsed dataset type
        self.data_definitions = {
            'images': {'size': [-1, self.sequence_length, 3, self.img_size, self.img_size], 'type': [torch.Tensor]},
            'tasks': {'size': [-1, 1], 'type': [list, str]},
            'questions': {'size': [-1, self.nwords], 'type': [torch.Tensor]},
            #'targets': {'size': [-1,self.sequence_length, self.output_classes], 'type': [torch.Tensor]},
            'targets_pointing': {'size': [-1, self.sequence_length, 2], 'type': [torch.Tensor]},
            'targets_answer': {'size': [-1, self.sequence_length, self.output_classes], 'type': [list, str]},
            'masks_pnt': {'size': [-1, self.sequence_length], 'type': [torch.Tensor]},
            'masks_word': {'size': [-1, self.sequence_length], 'type': [torch.Tensor]},
        }

        # Check if dataset exists, download or generate if necessary.
        self.source_dataset()

        # Load all the .jsons, but image generation is done in __getitem__
        self.dataset = []

        logging.info("Loading dataset as json into memory.")
        # Val and Test are not shuffled
        if self.set == 'val' or self.set == 'test':
            for tasklist in os.listdir(self.data_folder_child):
                if tasklist[4:-8] in self.tasks:
                    with gzip.open(os.path.join(self.data_folder_child, tasklist)) as f:
                        fulltask = f.read().decode('utf-8').split('\n')
                        for datapoint in fulltask:
                            self.dataset.append(json.loads(datapoint))
                    logging.info("{} task examples loaded.".format(tasklist[4:-8]))
                else:
                    logging.info("Skipped loading {} task.".format(tasklist[4:-8]))

        # Training set is shuffled
        elif self.set == 'train':
            for zipfile in os.listdir(self.data_folder_child):
                with gzip.open(os.path.join(self.data_folder_child, zipfile)) as f:
                    fullzip = f.read().decode('utf-8').split('\n')
                    for datapoint in fullzip:
                        task = json.loads(datapoint)
                        if task['family'] in self.tasks:
                            self.dataset.append(task)
                logging.info("Zip file {} loaded.".format(zipfile))

        self.length = len(self.dataset)

    @staticmethod
    def calculate_accuracy(data_dict, logits):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.
        WARNING: Applies mask to both logits and targets!

        :param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask'}).

        :param logits: Predictions being output of the model.

        """
        # Get targets.
        targets_answer = data_dict['targets_answer']
        targets_pointing = data_dict['targets_pointing']

        # Get predictions.
        preds_answer = logits[0]
        preds_pointing = logits[1]

        # Get sizes.
        batch_size = logits[0].size(0)
        img_seq_len = logits[0].size(1)

        # Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
        preds_answer = preds_answer.view(batch_size * img_seq_len, -1)
        preds_pointing = preds_pointing.view(batch_size * img_seq_len, -1)

        # Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
        targets_answer = targets_answer.view(batch_size * img_seq_len)
        # Reshape targets: pointings [BATCH_SIZE * IMG_SEQ_LEN x NUM_ACTIONS]
        targets_pointing = targets_pointing.view(batch_size * img_seq_len, -1)

        # Retrieve "answer" and "pointing" masks, both of size [BATCH_SIZE * IMG_SEQ_LEN].
        mask_answer = data_dict['masks_word']
        mask_answer = mask_answer.view(batch_size * img_seq_len)

        mask_pointing = data_dict['masks_pnt']
        mask_pointing = mask_pointing.view(batch_size * img_seq_len)

        # print("targets_answer = ", targets_answer)
        # print("preds_answer = ", preds_answer)
        # print("mask_answer = ", mask_answer)

        # print("targets_pointing = ", targets_pointing)
        # print("preds_pointing = ", preds_pointing)
        # print("mask_pointing = ", mask_pointing)

        #########################################################################
        # Calculate accuracy for Answering task.
        # Get answers [BATCH_SIZE * IMG_SEQ_LEN]
        _, indices = torch.max(preds_answer, 1)

        # Calculate correct answers with additional "masking".
        correct_answers = (indices == targets_answer).type(torch.ByteTensor) * mask_answer

        # Calculate accurary.
        if mask_answer.sum() > 0:
            acc_answer = float(correct_answers.sum().item()) / float(mask_answer.sum().item())
        else:
            acc_answer = 0.0

        #########################################################################
        # Calculate accuracy for Pointing task.

        # Normalize pointing with softmax.
        softmax_pointing = nn.Softmax(dim=1)
        preds_pointing = softmax_pointing(preds_pointing)

        # Calculate mean square error for every pointing action.
        diff_pointing = targets_pointing - preds_pointing
        diff_pointing = diff_pointing ** 2
        # Sum all differences for a given answer.
        # As a results we got 1D tensor of size [BATCH_SIZE * IMG_SEQ_LEN].
        diff_pointing = torch.sum(diff_pointing, dim=1)

        # Apply  threshold.
        threshold = 0.15 ** 2

        # Check correct pointings.
        correct_pointing = (diff_pointing < threshold).type(torch.ByteTensor) * mask_pointing
        # print('corect poitning',correct_pointing)
        # Calculate accurary.
        if mask_pointing.sum() > 0:
            acc_pointing = float(correct_pointing.sum().item()) / float(mask_pointing.sum().item())
        else:
            acc_pointing = 0.0

        #########################################################################
        # Total accuracy.
        acc_total = float(correct_answers.sum() + correct_pointing.sum()) / float(
            mask_answer.sum() + mask_pointing.sum()
        )
        # acc_total = torch.mean(torch.cat( (correct_answers.type(torch.FloatTensor), correct_pointing.type(torch.FloatTensor)) ) )

        # Return all three of them.
        return acc_total, acc_answer, acc_pointing

    @staticmethod
    def get_acc_per_family(data_dict, logits):
        """
        Compute the accuracy per family for the current batch. Also accumulates
        the number of correct predictions & questions per family in self.correct_pred_families (saved
        to file).


        .. note::

            To refactor.


        :param data_dict: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', \
        'targets', 'targets_string', 'index','imgfiles'})
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: network predictions.
        :type logits: :py:class:`torch.Tensor`

        """

        # Get targets.
        targets_answer = data_dict['targets_answer']
        targets_pointing = data_dict['targets_pointing']

        # build dictionary to store acc families stats
        tuple_list = [[0, 0, 0] for _ in range(len(constants.CATEGORIES))]
        categories_stats = dict(zip(constants.CATEGORIES, tuple_list))

        # Get tasks
        tasks = data_dict['tasks']

        # Get predictions.
        preds_answer = logits[0]
        preds_pointing = logits[1]

        # Get sizes.
        batch_size = logits[0].size(0)
        img_seq_len = logits[0].size(1)

        # Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
        preds_answer = preds_answer.view(batch_size * img_seq_len, -1)
        preds_pointing = preds_pointing.view(batch_size * img_seq_len, -1)

        # Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
        targets_answer = targets_answer.view(batch_size * img_seq_len)
        # Reshape targets: pointings [BATCH_SIZE * IMG_SEQ_LEN x NUM_ACTIONS]
        targets_pointing = targets_pointing.view(batch_size * img_seq_len, -1)

        # Retrieve "answer" and "pointing" masks, both of size [BATCH_SIZE * IMG_SEQ_LEN].
        mask_answer = data_dict['masks_word']
        mask_answer_non_flatten = mask_answer
        mask_answer = mask_answer.view(batch_size * img_seq_len)

        mask_pointing = data_dict['masks_pnt']
        mask_pointing_non_flatten = mask_pointing
        mask_pointing = mask_pointing.view(batch_size * img_seq_len)

        #########################################################################
        # Calculate accuracy for Answering task.
        # Get answers [BATCH_SIZE * IMG_SEQ_LEN]
        _, indices = torch.max(preds_answer, 1)

        # Calculate correct answers with additional "masking".
        correct_answers = (indices == targets_answer).type(torch.ByteTensor) * mask_answer

        #########################################################################
        # Calculate accuracy for Pointing task.

        # Normalize pointing with softmax.
        softmax_pointing = nn.Softmax(dim=1)
        preds_pointing = softmax_pointing(preds_pointing)

        # Calculate mean square error for every pointing action.
        diff_pointing = targets_pointing - preds_pointing
        diff_pointing = diff_pointing ** 2
        # Sum all differences for a given answer.
        # As a results we got 1D tensor of size [BATCH_SIZE * IMG_SEQ_LEN].
        diff_pointing = torch.sum(diff_pointing, dim=1)

        # Apply  threshold.
        threshold = 0.15 ** 2

        # Check correct pointings.
        correct_pointing = (diff_pointing < threshold).type(torch.ByteTensor) * mask_pointing

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

    def output_class_to_int(self, targets_answer):
        # for j, target in enumerate(targets_answer):
        targets_answer = [-1 if a == 'invalid' else self.output_vocab.index(a) for a in targets_answer]
        targets_answer = torch.LongTensor(targets_answer)
        return targets_answer

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'images', 'questions', 'targets', 'targets_label'})``, with:
        
            - ``images``: Sequence of images,
            - ``tasks``: Which task family sample belongs to,
            - ``questions``: Question on the sequence (this is constant per sequence for COG),
            - ``targets_pointing``: Sequence of targets as tuple of floats for pointing tasks,
            - ``targets_answer``: Sequence of word targets for classification tasks.

        """
        # This returns:
        # All variables are numpy array of float32
        # in_imgs: (n_epoch*batch_size, img_size, img_size, 3)
        # in_rule: (max_seq_length, batch_size) the rule language input, type int32
        # seq_length: (batch_size,) the length of each task instruction
        # out_pnt: (n_epoch*batch_size, n_out_pnt)
        # out_pnt_xy: (n_epoch*batch_size, -2)
        # out_word: (n_epoch*batch_size, n_out_word)
        # mask_pnt: (n_epoch*batch_size)
        # mask_word: (n_epoch*batch_size)

        # Get values from JSON.
        (in_imgs, _, _, out_pnt, _, _, mask_pnt, mask_word, _) = jti.json_to_feeds([self.dataset[index]])

        # Images [BATCH_SIZE x IMG_SEQ_LEN x DEPTH x HEIGHT x WIDTH].
        images = ((torch.from_numpy(in_imgs)).permute(1, 0, 4, 2, 3)).squeeze()
        images = images / 255  # Rescale RGB float from [0..255] to [0..1]

        # Set masks used in loss/accuracy calculations.
        mask_pnt = torch.from_numpy(mask_pnt).type(torch.ByteTensor)
        mask_word = torch.from_numpy(mask_word).type(torch.ByteTensor)

        tasks = self.dataset[index]['family']
        questions = [self.dataset[index]['question']]

        questions_string = [self.dataset[index]['question']]
        questions = torch.LongTensor([constants.INPUTVOCABULARY.index(word) for word in questions[0].split()])
        if questions.size(0) <= self.nwords:
            prev_size = questions.size(0)
            questions.resize_(self.nwords)
            questions[prev_size:] = 0

        # Set targets - depending on the answers.
        answers = self.dataset[index]['answers']
        answers_string = self.dataset[index]['answers']
        if tasks in constants.CLASSIFICATION_TASKS:
            targets_answer = self.output_class_to_int(answers)
        else:
            targets_answer = torch.LongTensor([-1 for target in answers])

        # Why are we always setting pointing targets, and answer targets only when required (-1 opposite)?
        targets_pointing = torch.FloatTensor(out_pnt)

        return (
            images,
            tasks,
            questions,
            targets_pointing,
            targets_answer,
            mask_pnt,
            mask_word,
            questions_string,
            answers_string,
        )

    @staticmethod
    def collate_fn(batch):
        """
        Combines a list of :py:class:`miprometheus.utils.DataDict` (retrieved with :py:func:`__getitem__`) into a batch.

        :param batch: individual :py:class:`miprometheus.utils.DataDict` samples to combine.
        :type batch: list

        :return: ``DataDict({'images', 'tasks', 'questions', 'targets_pointing', 'targets_answer'})`` containing the batch.

        """
        # Transpose the list of batches of 1
        # Using zip_longest to insert None in case of different length lists (shouldn't happen though)
        (
            images,
            tasks,
            questions,
            targets_pointing,
            targets_answer,
            mask_pnt,
            mask_word,
            questions_string,
            answers_string,
        ) = map(list, zip_longest(*batch))

        images = torch.stack(images).type(torch.FloatTensor)
        questions = torch.stack(questions).type(torch.LongTensor)
        # Targets.
        targets_pointing = torch.stack(targets_pointing).type(torch.FloatTensor)
        targets_answer = torch.stack(targets_answer).type(torch.LongTensor)
        # Masks.
        mask_pnt = torch.stack(mask_pnt).type(torch.ByteTensor)
        mask_word = torch.stack(mask_word).type(torch.ByteTensor)

        return (
            images,
            tasks,
            questions,
            targets_pointing,
            targets_answer,
            mask_pnt,
            mask_word,
            questions_string,
            answers_string,
        )

    def source_dataset(self):
        """
        Handles downloading and unzipping the canonical or hard version of the dataset.

        """
        self.download = False
        if self.dataset_type == 'generated':
            self.download = self.check_and_download(self.data_folder_child)
            if self.download:
                generate_dataset.main(
                    self.data_folder_parent,
                    self.examples_per_task,
                    self.sequence_length,
                    self.memory_length,
                    self.max_distractors,
                    self.nr_processors,
                )
                logging.info(f'\nDataset generation complete for {self.dataset_name}!')
                self.download = False

        if self.dataset_type == 'canonical':
            self.download = self.check_and_download(
                self.data_folder_child, 'https://storage.googleapis.com/cog-datasets/data_4_3_1.tar'
            )

        elif self.dataset_type == 'hard':
            self.download = self.check_and_download(
                self.data_folder_child, 'https://storage.googleapis.com/cog-datasets/data_8_7_10.tar'
            )
        if self.download:
            logging.info('\nDownload complete. Extracting...')
            tar = tarfile.open(os.path.expanduser('~/data/downloaded'))
            tar.extractall(path=self.data_folder_main)
            tar.close()
            logging.info('\nDone! Cleaning up.')
            os.remove(os.path.expanduser('~/data/downloaded'))
            logging.info('\nClean-up complete! Dataset ready.')

    def add_statistics(self, stat_col):
        """
        Add :py:class:`COG`-specific stats to :py:class:`miprometheus.utils.StatisticsCollector`.

        :param stat_col: :py:class:`miprometheus.utils.StatisticsCollector`.

        """
        for stat in ['loss_answer', 'loss_pointing', 'acc', 'acc_answer', 'acc_pointing'] + constants.CATEGORIES:
            stat_col.add_statistic(stat, '{:12.10f}')

    @staticmethod
    def collect_statistics(stat_col, data_dict, logits, loss_answer, loss_pointing):
        """
        Collects dataset details.
        :param stat_col: :py:class:`miprometheus.utils.StatisticsCollector`.
        :param data_dict: :py:class:`miprometheus.utils.DataDict` containing targets.
        :param logits: Prediction of the model (:py:class:`torch.Tensor`)
        """
        # Additional loss.
        stat_col['loss_answer'] = loss_answer.cpu().item()
        stat_col['loss_pointing'] = loss_pointing.cpu().item()

        # Accuracies.
        acc_total, acc_answer, acc_pointing = COG.calculate_accuracy(data_dict, logits)
        stat_col['acc'] = acc_total
        stat_col['acc_answer'] = acc_answer
        stat_col['acc_pointing'] = acc_pointing

        # Families Accuracies
        families_accuracies_dic = COG.get_acc_per_family(data_dict, logits)

        for key in families_accuracies_dic:
            stat_col[key] = families_accuracies_dic[key][2]

    @staticmethod
    def show_sample(batch, sample_number=0):
        """
        Shows a sample from the batch.

        :param data_dict: ``DataDict`` containing inputs and targets.
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param sample_number: Number of sample in batch (default: 0)
        :type sample_number: int
        """

        import matplotlib.pyplot as plt

        # Transpose, pick desired sample, unpack
        (
            images,
            tasks,
            questions,
            targets_pointing,
            targets_answer,
            mask_pnt,
            mask_word,
            questions_string,
            answers_string,
        ) = list(zip(*batch))[sample_number]

        # show data.
        print(
            f"sample_number={sample_number}",
            f"tasks={tasks}",
            f"questions={questions}",
            f"targets_answer={targets_answer}",
            f"mask_pnt={mask_pnt}",
            f"mask_word={mask_word}",
            f"questions_string={questions_string}",
            f"answers_string={answers_string}",
            sep='\n',
        )

        # Convert the images to numpy
        images = images.cpu().detach().numpy()
        targets_pointing = targets_pointing.cpu().detach().numpy()

        fig, axs = plt.subplots(nrows=2, ncols=len(images))
        for i, (image, target_pointing) in enumerate(zip(images, targets_pointing)):
            # Dims (c, h, w) -> (h, w, c)
            image = image.transpose(1, 2, 0)
            axs[0, i].imshow(image)

            # Reshape target_pointing
            target_pointing = target_pointing.reshape((constants.GRID_SIZE, constants.GRID_SIZE))
            axs[1, i].imshow(target_pointing)

        # Plot!
        plt.show()

    # Function to make check and download easier
    @staticmethod
    def check_and_download(file_folder_to_check, url=None, download_name='~/data/downloaded'):
        """
        Checks whether a file or folder exists at given path (relative to storage folder), \
        otherwise downloads files from the given URL.

        :param file_folder_to_check: Relative path to a file or folder to check to see if it exists.
        :type file_folder_to_check: str

        :param url: URL to download files from.
        :type url: str

        :param download_name: What to name the downloaded file. (DEFAULT: "downloaded").
        :type download_name: str

        :return: False if file was found, True if a download was necessary.

        """

        file_folder_to_check = os.path.expanduser(file_folder_to_check)
        if not (os.path.isfile(file_folder_to_check) or os.path.isdir(file_folder_to_check)):
            if url is not None:
                logging.info('Downloading {}'.format(url))
                wget.download(url, os.path.expanduser(download_name))
                return True
            else:
                return True
        else:
            logging.info('Dataset found at {}'.format(file_folder_to_check))
            return False


if __name__ == "__main__":

    """ 
    Unit test that checks data dimensions match expected values, and generates an image.
    Checks one regression and one classification task.
    """

    # Test parameters
    batch_size = 44

    # Timing test parameters
    timing_test = False
    testbatches = 100

    # -------------------------

    # Define useful params
    tasks = ['ExistColorGo']

    # Create problem - task Go
    cog_dataset = COG(data_folder='~/data/cog', subset='val', cog_type='canonical', cog_tasks=tasks)

    # Set up Dataloader iterator
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset=cog_dataset, collate_fn=cog_dataset.collate_fn, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # Display single sample (0) from batch.
    batch = next(iter(dataloader))

    # Show sample - Task 1
    cog_dataset.show_sample(batch, 0)

    # Show sample - Task 2
    cog_dataset.show_sample(batch, 1)

    if timing_test:
        # Test speed of generating images vs preloading generated images.
        import time

        # Define params to load entire dataset - all tasks included
        preload = time.time()
        full_cog_canonical = COG(data_folder='~/data/cog/', subset='val', cog_type='canonical', cog_tasks='all')
        postload = time.time()

        dataloader = DataLoader(
            dataset=full_cog_canonical,
            collate_fn=full_cog_canonical.collate_fn,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
        )

        prebatch = time.time()
        for i, batch in enumerate(dataloader):
            if i == testbatches:
                break
            if i % 100 == 0:
                print(f'Batch # {i} - {type(batch)}')
        postbatch = time.time()

        print(f'Number of workers: {dataloader.num_workers}')
        print(f'Time taken to load the dataset: {postload - preload}s')
        print(
            f'Time taken to exhaust {testbatches} batches for a batch size of {batch_size} with image generation: {postbatch - prebatch}s'
        )

        # Test pregeneration and loading
        for i, batch in enumerate(dataloader):
            if i == testbatches:
                print(f'Finished saving {testbatches} batches')
                break
            (
                images,
                tasks,
                questions,
                targets_pointing,
                targets_answer,
                mask_pnt,
                mask_word,
                questions_string,
                answers_string,
            ) = batch
            if not os.path.exists(os.path.expanduser('~/data/cogtest')):
                os.makedirs(os.path.expanduser('~/data/cogtest'))
            np.save(os.path.expanduser('~/data/cogtest/' + str(i)), images)

        preload = time.time()
        for i in range(testbatches):
            mockload = np.fromfile(os.path.expanduser('~/data/cogtest/' + str(i) + '.npy'))
        postload = time.time()
        print(
            f'Generation time for {testbatches} batches: {postbatch - prebatch}, Load time for {testbatches} batches: {postload - preload}'
        )

        print('Timing test completed, removing files.')
        for i in range(testbatches):
            os.remove(os.path.expanduser('~/data/cogtest/' + str(i) + '.npy'))

    print('Done!')
