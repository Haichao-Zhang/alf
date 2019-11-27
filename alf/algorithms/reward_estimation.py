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
from alf.utils.adaptive_normalizer import ScalarAdaptiveNormalizer
from alf.utils.encoding_network import EncodingNetwork

ICMInfo = namedtuple("ICMInfo", ["reward", "loss"])


@gin.configurable
class RewardAlgorithm(Algorithm):
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
                 fuse_net: Network = None,
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
            fuse_net (Network): from distance to reward.
        """
        super(RewardAlgorithm, self).__init__(
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

        # if fuse_net is None:
        #     fuse_net = EncodingNetwork(
        #         name="inverse_net",
        #         input_tensor_spec=[feature_spec, feature_spec],
        #         fc_layer_params=hidden_size,
        #         last_layer_size=self._num_actions,
        #         last_kernel_initializer=tf.initializers.Zeros())

        self._fuse_net = fuse_net

        self._reward_normalizer = ScalarAdaptiveNormalizer(
            speed=reward_adapt_speed)

    def _encode_action(self, action):
        if tensor_spec.is_discrete(self._action_spec):
            return tf.one_hot(indices=action, depth=self._num_actions)
        else:
            return action

    def train_step(self, inputs, state, calc_intrinsic_reward=True):
        """
        Args:
            inputs (tuple): observation and previous action
            state (Tensor):  state for ICM (previous feature)
            calc_intrinsic_reward (bool): if False, only return the losses
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: empty tuple ()
                info (ICMInfo):
        """
        inputs_obs, reward_external = inputs
        if self._encoding_net is not None:
            feature, _, goal_state = self._encoding_net(inputs_obs)

        #goal_state = inputs_obs[1]
        # goal_pos, goal_vol = tf.split(goal_state, num_or_size_splits=2, axis=1)
        self_pos = self_state  # all position based state now, no need to split

        # binary_mask = tf.dtypes.cast(
        #     tf.math.greater(reward_external, tf.zeros_like(reward_external)),
        #     tf.float32)
        # masked_reward = tf.multiply(binary_mask, reward_external)

        # print(binary_mask)
        # print(reward_external)
        # print(masked_reward)
        # forward_loss = 0.5 * tf.multiply(
        #     tf.reduce_mean(tf.square(feature - goal_pos), axis=-1),
        #     tf.stop_gradient(masked_reward))  # reduce the last dim

        # reward_pred = self._fuse_net(
        #     tf.reduce_mean(tf.square(feature - goal_pos), axis=-1))

        reward_pred, _ = self._fuse_net(feature - self_pos)
        # print("====pred")
        # print(reward_pred)
        # print("------")
        # print(reward_external)
        forward_loss = 0.5 * tf.square(
            reward_pred - reward_external)  # reduce the last dim

        intrinsic_reward = ()
        if calc_intrinsic_reward:
            # negative forward (pose estimation) loss as reward
            intrinsic_reward = tf.stop_gradient(
                reward_pred
            )  # can either use reward pred as reward of forward loss as reward
            # intrinsic_reward = tf.stop_gradient(
            #     -forward_loss
            # )  # can either use reward pred as reward of forward loss as reward
            # intrinsic_reward = self._reward_normalizer.normalize(
            #     intrinsic_reward)

        return AlgorithmStep(
            outputs=(),
            state=feature,
            info=ICMInfo(
                reward=intrinsic_reward,
                loss=LossInfo(
                    loss=forward_loss,
                    extra=dict(forward_loss=forward_loss, ))))

    def calc_loss(self, info: ICMInfo):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(scalar_loss=loss.loss, extra=loss.extra)
