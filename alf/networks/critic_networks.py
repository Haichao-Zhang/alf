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

import gin

import torch
import torch.nn as nn

import alf.layers as layers
import alf.nest as nest
from alf.networks import EncodingNetwork, LSTMEncodingNetwork
from alf.tensor_specs import TensorSpec
from .network import Network


@gin.configurable
class CriticNetwork(Network):
    """Create an instance of CriticNetwork."""

    def __init__(self,
                 input_tensor_spec,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_fc_layer_params=None,
                 joint_fc_layer_params=None,
                 activation=torch.relu):
        """Creates an instance of `CriticNetwork` for estimating action-value of
        continuous actions. The action-value is defined as the expected return
        starting from the given input observation and taking the given action.
        This module takes observation as input and action as input and outputs
        an action-value tensor with the shape of [batch_size].

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            observation_conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            observation_fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes for observations.
            action_fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes for actions.
            joint_fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.

        """
        super(CriticNetwork, self).__init__(input_tensor_spec, (), "")

        observation_spec, action_spec = input_tensor_spec

        flat_action_spec = nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')

        self._single_action_spec = flat_action_spec[0]
        self._obs_encoder = EncodingNetwork(
            observation_spec,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation)

        self._action_encoder = EncodingNetwork(
            action_spec,
            fc_layer_params=action_fc_layer_params,
            activation=activation)

        self._joint_encoder = EncodingNetwork(
            TensorSpec((self._obs_encoder.output_size +
                        self._action_encoder.output_size, )),
            fc_layer_params=joint_fc_layer_params,
            activation=activation,
            last_layer_size=1,
            last_activation=layers.identity)

    def forward(self, inputs, state=()):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state: empty for API consistent with CriticRNNNetwork

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size]
            state: empty
        """
        observations, actions = inputs
        actions = actions.to(torch.float32)

        encoded_obs = self._obs_encoder(observations)
        encoded_action = self._action_encoder(actions)
        joint = torch.cat([encoded_obs, encoded_action], -1)
        action_value = self._joint_encoder(joint)
        return torch.squeeze(action_value, -1), state

    @property
    def state_spec(self):
        return ()


@gin.configurable
class CriticRNNNetwork(Network):
    """Creates a critic network with RNN."""

    def __init__(self,
                 input_tensor_spec,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_fc_layer_params=None,
                 joint_fc_layer_params=None,
                 lstm_hidden_size=100,
                 post_rnn_fc_layer_params=None,
                 activation=torch.relu):
        """Creates an instance of `CriticRNNNetwork` for estimating action-value
        of continuous actions. The action-value is defined as the expected return
        starting from the given inputs (observation and state) and taking the
        given action. It takes observation and state as input and outputs an
        action-value tensor with the shape of [batch_size].

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            observation_conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            observation_fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes for observations.
            action_fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes for actions.
            joint_fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            lstm_hidden_size (int or list[int] or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            post_rnn_fc_layer_params (list[int]): a list of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.


        """
        super(CriticRNNNetwork, self).__init__(input_tensor_spec, (), "")

        observation_spec, action_spec = input_tensor_spec

        flat_action_spec = nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')

        self._single_action_spec = flat_action_spec[0]
        self._obs_encoder = EncodingNetwork(
            observation_spec,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation)

        self._action_encoder = EncodingNetwork(
            action_spec,
            fc_layer_params=action_fc_layer_params,
            activation=activation)

        self._joint_encoder = EncodingNetwork(
            TensorSpec((self._obs_encoder.output_size +
                        self._action_encoder.output_size, )),
            fc_layer_params=joint_fc_layer_params,
            activation=activation)

        self._lstm_encoding_net = LSTMEncodingNetwork(
            self._joint_encoder.output_size,
            lstm_hidden_size,
            post_rnn_fc_layer_params,
            activation,
            last_layer_size=1,
            last_activation=layers.identity)

    def forward(self, inputs, state=()):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size]
            new_state (nest[tuple]): the updated states
        """
        observations, actions = inputs
        actions = actions.to(torch.float32)

        encoded_obs = self._obs_encoder(observations)
        encoded_action = self._action_encoder(actions)
        joint = torch.cat([encoded_obs, encoded_action], -1)
        encoded_joint = self._joint_encoder(joint)
        action_value, state = self._lstm_encoding_net(encoded_joint, state)
        return torch.squeeze(action_value, -1), state

    @property
    def state_spec(self):
        return self._lstm_encoding_net.state_spec
