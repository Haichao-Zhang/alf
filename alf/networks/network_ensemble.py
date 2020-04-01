# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Network Ensembles"""

import gin
import functools
import math

import torch
import torch.nn as nn

import alf.utils.math_ops as math_ops
import alf.nest as nest
from alf.networks import Network, EncodingNetwork, LSTMEncodingNetwork
from alf.networks.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec


class Ensemble(Network):
    """
    Creates an ensemble of neural network functions with randomized prior
    functions (Osband et. al. 2018).
    """

    def __init__(self,
                 base_model: Network,
                 prior_model=None,
                 ens_size=1,
                 prior_scale=0.1,
                 kappa=0.1):
        self._ens_size = ens_size
        self._kappa = kappa
        self._prior_scale = prior_scale
        self._prior_model = prior_model
        self.models = nn.ModuleList()
        self.priors = nn.ModuleList()

        for i in range(self._ens_size):
            self.models.append(base_model.copy())
            if prior_model is not None:
                self.priors.append(prior_model.copy())

    def pforward(self, ind, x):
        pred_i = self.models[ind].forward(x)
        if self._prior_model is not None:
            prior = self.priors[ind].forward(x)
            pred_i = pred_i + self._prior_scale * prior.detach()
        return pred_i

    def get_preds(self, x):
        return [self.pforward(i, x) for i in range(len(self.models))]

    def forward(self, x):
        preds = [None] * self._ens_size
        for i in range(len(self.models)):
            preds[i] = self.pforward(i, x)

        if self._kappa is not None:
            # log n is correction for adding together multiple values
            n = torch.tensor(len(self.models), dtype=torch.float32)
            exp_term = self._kappa * preds - torch.log(n)
            lse = torch.logsumexp(exp_term, dim=1, keepdim=True)
            return (1 / self.kappa) * lse
        else:
            # hard max
            return torch.max(preds)
