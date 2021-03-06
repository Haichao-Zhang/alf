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
from collections import namedtuple

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.encoding_network import EncodingNetwork
from alf.data_structures import ActionTimeStep

ICMInfo = namedtuple("ICMInfo", ["reward", "loss"])


@gin.configurable
class ICMAlgorithm(Algorithm):
    """Intrinsic Curiosity Module

    This module generate the intrinsic reward based on predition error of
    observation.

    See Pathak et al "Curiosity-driven Exploration by Self-supervised Prediction"
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 reward_adapt_speed=8.0,
                 encoding_net: Network = None,
                 forward_net: Network = None,
                 inverse_net: Network = None,
                 name="ICMAlgorithm"):
        """Create an ICMAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            reward_adapt_speed (float): how fast to adapt the reward normalizer.
                rouphly speaking, the statistics for the normalization is
                calculated mostly based on the most recent T/speed samples,
                where T is the total number of samples.
            encoding_net (Network): network for encoding observation into a
                latent feature specified by feature_spec. Its input is same as
                the input of this algorithm.
            forward_net (Network): network for predicting next feature based on
                previous feature and action. It should accept input with spec
                [feature_spec, encoded_action_spec] and output a tensor of shape
                feature_spec. For discrete action, encoded_action is an one-hot
                representation of the action. For continuous action, encoded
                action is same as the original action.
            inverse_net (Network): network for predicting previous action given
                the previous feature and current feature. It should accept input
                with spec [feature_spec, feature_spec] and output tensor of
                shape (num_actions,).
        """
        super(ICMAlgorithm, self).__init__(
            train_state_spec=feature_spec, name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        assert len(
            flat_action_spec) == 1, "ICM doesn't suport nested action_spec"

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "ICM doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        if tensor_spec.is_discrete(action_spec):
            self._num_actions = action_spec.maximum - action_spec.minimum + 1
        else:
            self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec

        feature_dim = flat_feature_spec[0].shape[-1]

        self._encoding_net = encoding_net

        if isinstance(hidden_size, int):
            hidden_size = (hidden_size, )

        if forward_net is None:
            encoded_action_spec = tensor_spec.TensorSpec((self._num_actions, ),
                                                         dtype=tf.float32)
            forward_net = EncodingNetwork(
                name="forward_net",
                input_tensor_spec=[feature_spec, encoded_action_spec],
                fc_layer_params=hidden_size,
                last_layer_size=feature_dim)

        self._forward_net = forward_net

        if inverse_net is None:
            inverse_net = EncodingNetwork(
                name="inverse_net",
                input_tensor_spec=[feature_spec, feature_spec],
                fc_layer_params=hidden_size,
                last_layer_size=self._num_actions,
                last_kernel_initializer=tf.initializers.Zeros())

        self._inverse_net = inverse_net

        self._reward_normalizer = ScalarAdaptiveNormalizer(
            speed=reward_adapt_speed)

    def _encode_action(self, action):
        if tensor_spec.is_discrete(self._action_spec):
            return tf.one_hot(indices=action, depth=self._num_actions)
        else:
            return action

    def train_step(self,
                   time_step: ActionTimeStep,
                   state,
                   calc_intrinsic_reward=True):
        """
        Args:
            time_step (ActionTimeStep): input time_step data for ICM
            state (Tensor): state for ICM (previous observation)
            calc_intrinsic_reward (bool): if False, only return the losses
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: observation
                info (ICMInfo):
        """
        feature = time_step.observation
        prev_action = time_step.prev_action

        if self._encoding_net is not None:
            feature, _ = self._encoding_net(feature)
        prev_feature = state
        prev_action = self._encode_action(prev_action)

        forward_pred, _ = self._forward_net(
            inputs=[tf.stop_gradient(prev_feature), prev_action])
        forward_loss = 0.5 * tf.reduce_mean(
            tf.square(tf.stop_gradient(feature) - forward_pred), axis=-1)

        action_pred, _ = self._inverse_net(inputs=[prev_feature, feature])

        if tensor_spec.is_discrete(self._action_spec):
            inverse_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=prev_action, logits=action_pred)
        else:
            inverse_loss = 0.5 * tf.reduce_mean(
                tf.square(prev_action - action_pred), axis=-1)

        intrinsic_reward = ()
        if calc_intrinsic_reward:
            intrinsic_reward = tf.stop_gradient(forward_loss)
            intrinsic_reward = self._reward_normalizer.normalize(
                intrinsic_reward)

        return AlgorithmStep(
            outputs=(),
            state=feature,
            info=ICMInfo(
                reward=intrinsic_reward,
                loss=LossInfo(
                    loss=forward_loss + inverse_loss,
                    extra=dict(
                        forward_loss=forward_loss,
                        inverse_loss=inverse_loss))))

    def calc_loss(self, info: ICMInfo):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(scalar_loss=loss.loss, extra=loss.extra)
