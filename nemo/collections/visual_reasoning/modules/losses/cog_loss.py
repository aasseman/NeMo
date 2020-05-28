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

from nemo.backends.pytorch import LogitsType, LossNM, LossType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs


class COGLoss(LossNM):
    def __init__(self):
        super().__init__()

    @property
    @add_port_docs()
    def input_ports(self):
        """ Returns definitions of module input ports. """
        return {
            "prediction_answers": NeuralType(axes=('B'), elements_type=LogitsType()),
            "prediction_pointings": NeuralType(axes=('B'), elements_type=LogitsType()),
            "target_answers": NeuralType(axes=('B'), elements_type=LogitsType()),
            "target_pointings": NeuralType(axes=('B'), elements_type=LogitsType()),
            "mask_pointers": NeuralType(axes=('B'), elements_type=MaskType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """ Returns definitions of module output ports. """
        return {
            "loss": NeuralType(elements_type=LossType()),
            "loss_answer": NeuralType(elements_type=LossType()),
            "loss_pointing": NeuralType(elements_type=LossType()),
        }

    # You need to implement this function
    def _loss_function(
        self, prediction_answers, prediction_pointings, target_answers, target_pointings, mask_pointers
    ):
        """Calculates accuracy equal to mean number of correct predictions in a given batch.
        The function calculates two separate losses for answering and pointing actions and sums them up.
        """

        # Get sizes.
        batch_size = prediction_answers.size(0)
        img_seq_len = prediction_answers.size(1)

        # Retrieve "pointing" masks, both of size [BATCH_SIZE x IMG_SEQ_LEN] and transform it into floats.
        mask_pointing = mask_pointers.type(torch.FloatTensor)

        # Classification loss.
        # Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
        prediction_answers = prediction_answers.view(batch_size * img_seq_len, -1)
        # Reshape targets [BATCH_SIZE * IMG_SEQ_LEN]
        target_answers = target_answers.view(batch_size * img_seq_len)
        # Calculate loss.
        # Ignore_index: specifies a target VALUE that is ignored and does not contribute to the input gradient.
        # -1 is set when we do not use that action.
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_answer = ce_loss_fn(prediction_answers, target_answers)

        # Pointing loss.
        # We will softmax over the third dimension of [BATCH_SIZE x IMG_SEQ_LEN x NUM_POINT_ACTIONS].
        logsoftmax_fn = nn.LogSoftmax(dim=2)
        # Calculate cross entropy [BATCH_SIZE x IMG_SEQ_LEN].
        ce_point = torch.sum((-target_pointings * logsoftmax_fn(prediction_pointings)), dim=2) * mask_pointing
        # print("mask_pointing =", mask_pointing)
        # print("ce_point = ", ce_point)

        # Calculate mean - manually, skipping all non-pointing elements of the targets.
        if mask_pointing.sum().item() != 0:
            self.loss_pointing = torch.sum(ce_point) / mask_pointing.sum()
        else:
            self.loss_pointing = torch.tensor(0).type(torch.FloatTensor)

        # Both losses are averaged over batch size and sequence lengts - so we can simply sum them.
        return self.loss_answer + self.loss_pointing, self.loss_answer, self.loss_pointing
