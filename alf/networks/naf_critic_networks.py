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
"""CriticNetworks"""

import gin
import functools
import math

import torch
import torch.nn as nn

import alf.layers as layers
import alf.nest as nest
from alf.networks import Network, EncodingNetwork, LSTMEncodingNetwork
from alf.networks.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec
from alf.utils import spec_utils
import alf.utils.math_ops as math_ops


@gin.configurable
class NafCriticNetwork(Network):
    """Create an instance of NafCriticNetwork."""

    def __init__(self,
                 input_tensor_spec,
                 observation_input_processors=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 mu_fc_layer_params=None,
                 v_fc_layer_params=None,
                 l_fc_layer_params=None,
                 activation=torch.tanh,
                 projection_output_init_gain=0.3,
                 std_bias_initializer_value=0.0,
                 kernel_initializer=None,
                 use_last_kernel_initializer=True,
                 last_activation=math_ops.identity,
                 name="CriticNetwork"):
        """Creates an instance of `NafCriticNetwork` for estimating action-value of
        continuous actions. The action-value is defined as the expected return
        starting from the given input observation and taking the given action.
        This module takes observation as input and action as input and outputs
        an action-value tensor with the shape of [batch_size].

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            observation_input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding observation input.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for actions.
            joint_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            name (str):
        """
        super().__init__(
            input_tensor_spec, skip_input_preprocessing=True, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        observation_spec, action_spec = input_tensor_spec

        flat_action_spec = nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')

        self._single_action_spec = flat_action_spec[0]
        action_dim = action_spec.shape[0]

        # [obs_dim -> hidden_dim]
        self._obs_encoder = EncodingNetwork(
            observation_spec,
            input_preprocessors=observation_input_processors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        if use_last_kernel_initializer:
            last_kernel_initializer = functools.partial(
                torch.nn.init.uniform_, a=-0.003, b=0.003)
        else:
            last_kernel_initializer = None

        # [hidden_dim -> action_dim]
        # might need action squash
        # self._mu = EncodingNetwork(TensorSpec(
        #     (self._obs_encoder.output_spec.shape[0], )),
        #                            fc_layer_params=mu_fc_layer_params,
        #                            activation=activation,
        #                            kernel_initializer=kernel_initializer,
        #                            last_layer_size=action_dim,
        #                            last_activation=torch.tanh,
        #                            last_kernel_initializer=None)

        self._mu = layers.FC(
            self._obs_encoder.output_spec.shape[0],
            action_dim,
            activation=math_ops.identity,
            kernel_init_gain=projection_output_init_gain)

        self._L = layers.FC(
            self._obs_encoder.output_spec.shape[0],
            action_dim**2,
            activation=math_ops.identity,
            kernel_init_gain=projection_output_init_gain,
            bias_init_value=std_bias_initializer_value)

        # [hidden_dim -> 1]
        # TODO joint_fc_layer_params change to non-shared
        self._V = EncodingNetwork(
            TensorSpec((self._obs_encoder.output_spec.shape[0], )),
            fc_layer_params=v_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer)

        # self._L = EncodingNetwork(TensorSpec(
        #     (self._obs_encoder.output_spec.shape[0], )),
        #                           fc_layer_params=l_fc_layer_params,
        #                           activation=activation,
        #                           kernel_initializer=kernel_initializer,
        #                           last_layer_size=action_dim**2,
        #                           last_activation=math_ops.identity,
        #                           last_kernel_initializer=None)

        self._tril_mask = torch.tril(
            torch.ones(action_dim, action_dim), diagonal=-1).unsqueeze(0)
        self._diag_mask = torch.diag(
            torch.diag(torch.ones(action_dim, action_dim))).unsqueeze(0)

        self._output_spec = TensorSpec(())

    def forward(self, inputs, state=()):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state: empty for API consistent with CriticRNNNetwork

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size]
            state: empty
        """
        # this line is just a placeholder doing nothing
        inputs, state = Network.forward(self, inputs, state)

        observations, actions = inputs

        # 0 encode observation
        encoded_obs, _ = self._obs_encoder(observations)

        # 1 mu
        mu = self._mu(encoded_obs)
        #mu = spec_utils.scale_to_spec(mu.tanh(), self._single_action_spec)

        # 2 V
        V, _ = self._V(encoded_obs)

        # 3 Q
        Q = None
        if actions is not None:
            actions = actions.to(torch.float32)
            num_outputs = mu.size(1)
            L = self._L(encoded_obs)
            L = L.view(-1, num_outputs, num_outputs)
            L = L * \
                self._tril_mask.expand_as(
                    L) + math_ops.clipped_exp(L) * self._diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (actions - mu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return (mu, Q, V), state
