# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample Keras Value Network.

Implements a network that will generate the following layers:

  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # fc_layer_params
  Dense -> 1         # Value output
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils

import gin.tf


@gin.configurable
class MultiModalValueNetwork(network.Network):
    """Feed Forward value network. Reduces to 1 value output per batch item."""

    def __init__(self,
                 input_tensor_spec,
                 fc_layer_params_visual=(200, 100),
                 fc_layer_params_state=(200, 100),
                 fc_layer_params_fusion=(200, 100),
                 dropout_layer_params=None,
                 conv_layer_params_visual=None,
                 conv_layer_params_state=None,
                 activation_fn=tf.keras.activations.relu,
                 name='ValueNetwork'):
        """Creates an instance of `ValueNetwork`.

    Network supports calls with shape outer_rank + observation_spec.shape. Note
    outer_rank must be at least 1.

    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
        representing the input observations.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent', if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      name: A string representing name of the network.

    Raises:
      ValueError: If input_tensor_spec is not an instance of network.InputSpec.
      ValueError: If `input_tensor_spec.observations` contains more than one
      observation.
    """
        super(MultiModalValueNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        # if len(tf.nest.flatten(input_tensor_spec)) > 1:
        #     raise ValueError(
        #         'Network only supports observation specs with a single observation.'
        #     )

        self._postprocessing_layers_visual = utils.mlp_layers(
            conv_layer_params_visual,
            fc_layer_params_visual,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.
            glorot_uniform(),
            dropout_layer_params=dropout_layer_params,
            name='input_mlp_visual')

        self._postprocessing_layers_state = utils.mlp_layers(
            conv_layer_params_state,
            fc_layer_params_state,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.
            glorot_uniform(),
            dropout_layer_params=dropout_layer_params,
            name='input_mlp_state')

        self._postprocessing_layers_fusion = utils.mlp_layers(
            None,  # no cnn
            fc_layer_params_fusion,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.
            glorot_uniform(),
            dropout_layer_params=dropout_layer_params,
            name='multi_modal_mlp_fusion')

        self._postprocessing_layers_fused = []
        self._postprocessing_layers_fused.append(
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.compat.v1.initializers.random_uniform(
                    minval=-0.03, maxval=0.03),
            ))

    def call(self, observations, step_type=None, network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations,
                                               self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)

        observations = tf.nest.flatten(observations)
        # do multi-modality fusion here
        v_states = tf.cast(observations[0], tf.float32)
        s_states = tf.cast(observations[1], tf.float32)

        v_states = batch_squash.flatten(v_states)
        s_states = batch_squash.flatten(s_states)

        for layer in self._postprocessing_layers_visual:
            v_states = layer(v_states)
        for layer in self._postprocessing_layers_state:
            s_states = layer(s_states)

        #states = v_states + s_states

        states = tf.concat([v_states, s_states], axis=1)
        # print(states.shape)
        for layer in self._postprocessing_layers_fusion:
            states = layer(states)

        for layer in self._postprocessing_layers_fused:
            states = layer(states)

        value = tf.reshape(states, [-1])
        print("value ------")
        print(value)
        value = batch_squash.unflatten(value)
        return value, network_state
