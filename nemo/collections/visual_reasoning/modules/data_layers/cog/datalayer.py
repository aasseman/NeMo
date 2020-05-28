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
import tarfile
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import wget
from torch.utils.data import Dataset

from . import constants, generate_dataset
from . import json_to_img as jti
from nemo.backends.pytorch import DataLayerNM
from nemo.core import DeviceType, NeuralModuleFactory
from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, VoidType
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs


class COGDataLayer(DataLayerNM, Dataset):
    """
    Inherits :class:`nemo.backends.pytorch.DataLayerNM` and :class:`torch.utils.data.Dataset`.
    The COG dataset is a sequential VQA dataset.

    Inputs are a sequence of images of simple shapes and characters on a black background, \
    and a question based on these objects that relies on memory which has to be answered at every step of the sequence.

    See https://arxiv.org/abs/1803.06092 (`A Dataset and Architecture for Visual Reasoning with a Working Memory`)\
    for the reference paper.

    """

    def __init__(
        self,
        data_folder: str = '~/data/cog',
        subset: str = 'train',
        cog_tasks: str = 'class',
        cog_type: str = 'canonical',
        cog_gen_examples_per_task=None,
        cog_gen_sequence_length=None,
        cog_gen_memory_length=None,
        cog_gen_max_distractors=None,
        cog_gen_threads=1,
    ):
        """
        Initializes :class:`COGDataLayer`.

        :param data_folder: Folder to save to / load from the COG dataset. Defaults to ``"~/data/cog"``.
        :type data_folder: str, optional
        :param subset: Select which of training, validation, or test split to load. Has to be one of ``"train"``, ``"val"`` or ``"test"``.
            Defaults to ``"train"``.
        :type subset: str, optional
        :param cog_tasks: Select which COG tasks to load. Has to be one of ``'class'``, ``'reg'``, ``'all'``,
            ``'binary'``, or a fine-grained list of tasks. The complete list of COG tasks is in 
            :mod:`nemo.collections.visual_reasoning.modules.data_layers.cog.cog_utils.constants`'s `CATEGORIES`.
            Defaults to ``'class'``.
        :type cog_tasks: str or list(str), optional
        :param cog_type: Type of COG dataset. One of ``'canonical'``, ``'hard'`` or ``'generated'``.
            If ``'generated'``, the COG dataset will be generated programmatically using the settings given in the next arguments, ie.
            `cog_gen_examples_per_task`, `cog_gen_sequence_length`, `cog_gen_memory_length`, `cog_gen_max_distractors`, `cog_gen_threads`.
            Defaults to ``'canonical'``.
        :type cog_type: str, optional
        :param cog_gen_examples_per_task: Setting for the COG generator. Number of samples to generate per task,
            Defaults to None.
        :type cog_gen_examples_per_task: int or None, optional
        :param cog_gen_sequence_length: Setting for the COG generator. Number of frames in each sample.
            Defaults to None.
        :type cog_gen_sequence_length: int or None, optional
        :param cog_gen_memory_length: Setting for the COG generator. Memory span, in number of frames.
            Defaults to None.
        :type cog_gen_memory_length: int or None, optional
        :param cog_gen_max_distractors: Setting for the COG generator. Number of distractor objects,
            Defaults to None.
        :type cog_gen_max_distractors: int or None, optional
        :param cog_gen_threads: Setting for the COG generator. Number of CPU threads to use during dataset generation.
            Defaults to 1.
        :type cog_gen_threads: int, optional
        """

        # Call base class constructors
        super(COGDataLayer, self).__init__()

        # Data folder main is /path/cog
        # Data folder parent is data_X_Y_Z
        # Data folder children are train_X_Y_Z, test_X_Y_Z, or val_X_Y_Z
        self.data_folder_main = os.path.expanduser(data_folder)

        self.set = subset
        if self.set not in [
            'val',
            'test',
            'train',
        ]:
            raise ValueError(f"subset must be one of 'val', 'test', or 'train', got {self.set}")

        self.dataset_type = cog_type
        if self.dataset_type not in ['canonical', 'hard', 'generated']:
            raise ValueError(
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

        # Check if dataset exists, download or generate if necessary.
        self.source_dataset()

        # Load all the .jsons, but image generation is done in __getitem__
        self._dataset = []

        logging.info("Loading dataset as json into memory.")
        # Val and Test are not shuffled
        if self.set == 'val' or self.set == 'test':
            for tasklist in os.listdir(self.data_folder_child):
                if tasklist[4:-8] in self.tasks:
                    with gzip.open(os.path.join(self.data_folder_child, tasklist)) as f:
                        fulltask = f.read().decode('utf-8').split('\n')
                        for datapoint in fulltask:
                            self._dataset.append(json.loads(datapoint))
                    logging.info(f"{tasklist[4:-8]} task examples loaded.")
                else:
                    logging.info(f"Skipped loading {tasklist[4:-8]} task.")

        # Training set is shuffled
        elif self.set == 'train':
            for zipfile in os.listdir(self.data_folder_child):
                with gzip.open(os.path.join(self.data_folder_child, zipfile)) as f:
                    fullzip = f.read().decode('utf-8').split('\n')
                    for datapoint in fullzip:
                        task = json.loads(datapoint)
                        if task['family'] in self.tasks:
                            self._dataset.append(task)
                logging.info(f"Zip file {zipfile} loaded.")

        self.length = len(self._dataset)

    @add_port_docs
    @property
    def output_ports(self):
        """
        Creates definitions of output ports.
        """
        return {
            "images": NeuralType(('B', 'T', 'C', 'H', 'W'), elements_type=ChannelType),
            "tasks": NeuralType(('B'), elements_type=VoidType),
            "questions": NeuralType(('B', 'A'), elements_type=CategoricalValuesType),
            "targets_pointing": NeuralType(('B', 'T', 'A'), elements_type=CategoricalValuesType),
            "targets_answer": NeuralType(('B', 'T'), elements_type=CategoricalValuesType),
            "mask_pointing": NeuralType(('B', 'T'), elements_type=MaskType),
            "mask_word": NeuralType(('B', 'T'), elements_type=MaskType),
            "questions_string": NeuralType(('B'), elements_type=VoidType),
            "answers_string": NeuralType(('B', 'T'), elements_type=VoidType),
        }

    def __len__(self):
        return self.length

    @property
    def dataset(self):
        return self

    @property
    def data_iterator(self):
        return None

    def output_class_to_int(self, targets_answer):
        # for j, target in enumerate(targets_answer):
        targets_answer = [-1 if a == 'invalid' else self.output_vocab.index(a) for a in targets_answer]
        targets_answer = torch.LongTensor(targets_answer)
        return targets_answer

    def __getitem__(self, index):
        """Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int
        :return: as defined in `output_ports`.
        :rtype: tuple
        """

        # Get values from JSON.
        (in_imgs, _, _, out_pnt, _, _, mask_pnt, mask_word, _) = jti.json_to_feeds([self._dataset[index]])

        # Images [BATCH_SIZE x IMG_SEQ_LEN x DEPTH x HEIGHT x WIDTH].
        images = ((torch.from_numpy(in_imgs)).permute(1, 0, 4, 2, 3)).squeeze()
        images = images / 255  # Rescale RGB float from [0..255] to [0..1]

        # Set masks used in loss/accuracy calculations.
        mask_pnt = torch.from_numpy(mask_pnt).type(torch.ByteTensor)
        mask_word = torch.from_numpy(mask_word).type(torch.ByteTensor)

        tasks = self._dataset[index]['family']
        questions = [self._dataset[index]['question']]

        questions_string = [self._dataset[index]['question']]
        questions = torch.LongTensor([constants.INPUTVOCABULARY.index(word) for word in questions[0].split()])
        if questions.size(0) <= self.nwords:
            prev_size = questions.size(0)
            questions.resize_(self.nwords)
            questions[prev_size:] = 0

        # Set targets - depending on the answers.
        answers = self._dataset[index]['answers']
        answers_string = self._dataset[index]['answers']
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
        """Collate separate samples from `__getitem__` into a batch. Effectively doing a transpose of the input data.

        :param batch: List of samples.
        :type batch: list(tuple)
        :return: Tuple of Batches (one per output port)
        :rtype: tuple
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
        """Handles downloading and unzipping the canonical or hard version of the dataset."""

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

    @staticmethod
    def show_sample(batch, sample_number=0):
        """Shows a sample from the batch. Mainly for testing purposes.
        If a display is present, will plot the visuals with :mod:`matplotlib`.

        :param batch: Batch from a :class:`torch.utils.data.Dataloader` operating on an instance of :class:`COGDataLayer`.
        :param sample_number: Index of sample to show from the batch, defaults to 0
        :type sample_number: int, optional
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
                logging.info(f'Downloading {url}')
                wget.download(url, os.path.expanduser(download_name))
                return True
            else:
                return True
        else:
            logging.info(f'Dataset found at {file_folder_to_check}')
            return False


if __name__ == "__main__":

    """ 
    Unit test that checks data dimensions match expected values, and generates an image.
    Checks one regression and one classification task.
    """

    neural_factory = NeuralModuleFactory(log_dir='logs', create_tb_writer=False, placement=DeviceType.CPU)

    # Test parameters
    batch_size = 44

    # Timing test parameters
    timing_test = False
    testbatches = 100

    # -------------------------

    # Define useful params
    tasks = ['ExistColorGo']

    # Create problem - task Go
    cog_dataset = COGDataLayer(data_folder='~/data/cog', subset='val', cog_type='canonical', cog_tasks=tasks)

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
        full_cog_canonical = COGDataLayer(
            data_folder='~/data/cog/', subset='val', cog_type='canonical', cog_tasks='all'
        )
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
