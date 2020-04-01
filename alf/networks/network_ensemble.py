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
from alf.utils import tensor_utils
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
        super(Ensemble,
              self).__init__(input_tensor_spec=base_model._input_tensor_spec)

        self._input_tensor_spec = base_model._input_tensor_spec
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

    def pforward(self, ind, x, state):
        pred_i, state = self.models[ind].forward(x, state)
        if self._prior_model is not None:
            # TODO: how to incorporate state_p?
            prior, state_p = self.priors[ind].forward(x, state)
            pred_i = pred_i + self._prior_scale * prior.detach()
        return pred_i, state

    def get_preds(self, x, states):
        preds = [None] * self._ens_size
        states_new = [None] * self._ens_size
        for i in range(self._ens_size):
            preds[i], states_new[i] = self.pforward(i, x, states[i])
        return preds, states_new

    def get_preds_min(self, x, states=None):
        if states is None:
            states = [None] * self._ens_size
        preds, states_new = self.get_preds(x, states)
        pred_min = tensor_utils.list_min(preds)
        return pred_min, states_new

    def get_preds_max(self, x, states=None):
        if states is None:
            states = [None] * self._ens_size
        preds, states_new = self.get_preds(x, states)
        pred_max = tensor_utils.list_max(preds)
        return pred_max, states_new

    def forward(self, x, states):
        preds = [None] * self._ens_size
        states_new = [None] * self._ens_size
        for i in range(self._ens_size):
            preds[i], states_new[i] = self.pforward(i, x, states[i])

        if self._kappa is not None:
            # log n is correction for adding together multiple values
            n = torch.tensor(self._ens_size, dtype=torch.float32)
            exp_term = self._kappa * preds - torch.log(n)
            lse = torch.logsumexp(exp_term, dim=1, keepdim=True)
            return (1 / self.kappa) * lse, states_new
        else:
            # hard max
            return torch.max(preds), states_new
