# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import tensorflow as tf
from tf_agents.networks.encoding_network import EncodingNetwork as TFAEncodingNetwork
from alf.layers import NestConcatenate
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
import gin.tf


@gin.configurable
class MultiModalEncodingNetwork(TFAEncodingNetwork):
    """Feed Forward network with CNN and FNN layers.."""

    def __init__(self,
                 input_tensor_spec,
                 last_layer_size,
                 last_activation_fn=None,
                 fc_layer_params_visual=(200, 100),
                 fc_layer_params_state=(200, 100),
                 fc_layer_params_fusion=(200, 100),
                 dropout_layer_params=None,
                 conv_layer_params_visual=None,
                 conv_layer_params_state=None,
                 activation_fn=tf.keras.activations.relu,
                 dtype=tf.float32,
                 last_kernel_initializer=None,
                 last_bias_initializer=tf.initializers.Zeros(),
                 preprocessing_combiner=NestConcatenate(axis=-1),
                 **xargs):
        """Create an EncodingNetwork

        This EncodingNetwork allows the last layer to have different setting
        from the other layers.

        Args:
            last_layer_size (int): size of the last layer
            last_activation_fn: Activation function of the last layer.
            last_kernel_initializer: Initializer for the kernel of the last
                layer. If none is provided a default
                tf.initializers.VarianceScaling is used.
            last_bias_initializer: initializer for the bias of the last layer.
            preprocessing_combiner: (Optional.) A keras layer that takes a flat
                list of tensors and combines them. Good options include
                `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
                This layer must not be already built. For more details see
                the documentation of `networks.EncodingNetwork`. If there is only
                one input, this will be ignored.
            xargs (dict): See tf_agents.networks.encoding_network.EncodingNetwork
              for detail
        """
        if len(tf.nest.flatten(input_tensor_spec)) == 1:
            preprocessing_combiner = None
        super(MultiModalEncodingNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            preprocessing_combiner=preprocessing_combiner,
            dtype=dtype,
            **xargs)

        if not last_kernel_initializer:
            last_kernel_initializer = tf.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal')

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

        self._last_layer = tf.keras.layers.Dense(
            last_layer_size,
            activation=last_activation_fn,
            kernel_initializer=last_kernel_initializer,
            bias_initializer=last_bias_initializer,
            dtype=dtype)

    def call(self, observations, step_type=None, network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations,
                                               self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)

        observations = tf.nest.flatten(observations)
        # do multi-modality fusion here
        v_states = tf.cast(observations[0], tf.float32)
        s_states = tf.cast(observations[1], tf.float32)

        v_states = batch_squash.flatten(v_states)
        # s_states = batch_squash.flatten(s_states)

        for layer in self._postprocessing_layers_visual:
            v_states = layer(v_states)

        for layer in self._postprocessing_layers_state:
            s_states = layer(s_states)

        #states = v_states + s_states

        #states = tf.concat([v_states, s_states], axis=1)
        # for pose estimation, using v_states only
        #keep only the visual input
        states = v_states

        # print(states.shape)
        for layer in self._postprocessing_layers_fusion:
            states = layer(states)

        #states = tf.reshape(states, [-1])

        states = batch_squash.unflatten(states)

        self_state = s_states
        return self._last_layer(states), network_state, self_state
