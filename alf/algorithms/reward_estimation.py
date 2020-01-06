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
                 inverse_net: Network = None,
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

        print(feature_spec)

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

        self.feature_dim = feature_dim

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
        feature, prev_action, reward_external = inputs

        prev_action = tf.math.floormod(prev_action, 3.142)  # mod operation

        if self._encoding_net is not None:
            goal_feature, _, self_state_feature = self._encoding_net(feature)

        prev_feature = state
        prev_action = self._encode_action(prev_action)

        # the reward estimation module should estimate next state and reward jointly (a complete forward model)

        # the encoding net if not learned
        forward_pred, _ = self._forward_net(
            inputs=[tf.stop_gradient(prev_feature), prev_action])

        forward_loss = 0.5 * tf.reduce_mean(
            tf.square(tf.stop_gradient(self_state_feature) - forward_pred),
            axis=-1)

        action_pred, _ = self._inverse_net(
            inputs=[prev_feature, self_state_feature])

        if tensor_spec.is_discrete(self._action_spec):
            inverse_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=prev_action, logits=action_pred)
        else:
            inverse_loss = 0.5 * tf.reduce_mean(
                tf.square(prev_action - action_pred), axis=-1)
            # print(prev_action)
            # print(action_pred)
            # print("inverse")
            # print(inverse_loss)

        # reward prediction loss based on forward_pred
        self_pose = self_state_feature  # all position based state now, no need to split

        # reward estimation based on one-step forward prediction
        reward_input_feature = tf.concat(
            [
                tf.stop_gradient(forward_pred),  # stop gradient
                goal_feature,  # train goal encoding
            ],
            axis=1)

        reward_pred, _ = self._fuse_net(reward_input_feature)

        binary_mask = tf.dtypes.cast(
            tf.math.greater(reward_external, tf.zeros_like(reward_external)),
            tf.float32)

        scaled_mask = binary_mask * 1 + 1e-3 * (1 - binary_mask)

        #masked_reward = tf.multiply(scaled_mask, reward_external)

        pred_reward_mse = 0.5 * tf.square(
            reward_pred - reward_external)  # reduce the last dim
        reward_loss = 0. * tf.multiply(
            pred_reward_mse,
            tf.stop_gradient(scaled_mask))  # reduce the last dim

        #goal_state = inputs_obs[1]
        # goal_pos, goal_vol = tf.split(goal_state, num_or_size_splits=2, axis=1)
        #--------------------
        # self_pose = goal_state  # all position based state now, no need to split

        # binary_mask = tf.dtypes.cast(
        #     tf.math.greater(reward_external, tf.zeros_like(reward_external)),
        #     tf.float32)
        # masked_reward = tf.multiply(binary_mask, reward_external)
        # pred_mse = tf.reduce_mean(tf.square(feature - self_pose), axis=-1)
        # forward_loss = 0.5 * tf.multiply(
        #     pred_mse, tf.stop_gradient(masked_reward))  # reduce the last dim
        # reward_pred = -pred_mse
        #---------------------

        # reward_pred = self._fuse_net(
        #     tf.reduce_mean(tf.square(feature - goal_pos), axis=-1))

        # reward_pred, _ = self._fuse_net(feature - self_pos)
        # print(inputs_obs)
        # print(self_pose)
        # print("====pred")
        # print(reward_pred)
        # # print("------")
        # # print(reward_external)
        # forward_loss = 0.5 * tf.square(
        #     reward_pred - reward_external)  # reduce the last dim

        # intrinsic_reward = ()
        # if calc_intrinsic_reward:
        #     # negative forward (pose estimation) loss as reward
        #     intrinsic_reward = tf.stop_gradient(
        #         reward_pred
        #     )  # can either use reward pred as reward of forward loss as reward
        #     # intrinsic_reward = tf.stop_gradient(
        #     #     -forward_loss
        #     # )  # can either use reward pred as reward of forward loss as reward
        #     # intrinsic_reward = self._reward_normalizer.normalize(
        #     #     intrinsic_reward)

        intrinsic_reward = ()
        if calc_intrinsic_reward:
            # intrinsic_reward = tf.stop_gradient(forward_loss)
            intrinsic_reward = tf.stop_gradient(reward_pred)
            # intrinsic_reward = self._reward_normalizer.normalize(
            #     intrinsic_reward)

        return AlgorithmStep(
            outputs=(),
            state=self_state_feature,
            info=ICMInfo(
                reward=intrinsic_reward,
                loss=LossInfo(
                    loss=forward_loss + inverse_loss + reward_loss,
                    extra=dict(
                        forward_loss=forward_loss,
                        inverse_loss=inverse_loss,
                        reward_loss=reward_loss))))

    def calc_loss(self, info: ICMInfo):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(scalar_loss=loss.loss, extra=loss.extra)

    #=================Pure state based reward estimation


@gin.configurable
class RewardAlgorithmState(Algorithm):
    """Intrinsic Curiosity Module

    This module generate the intrinsic reward based on predition error of
    observation.

    See Pathak et al "Curiosity-driven Exploration by Self-supervised Prediction"
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 observation_spec,
                 hidden_size=256,
                 reward_adapt_speed=8.0,
                 encoding_net: Network = None,
                 forward_net: Network = None,
                 inverse_net: Network = None,
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
        super(RewardAlgorithmState, self).__init__(
            train_state_spec=feature_spec, name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        assert len(
            flat_action_spec) == 1, "ICM doesn't suport nested action_spec"

        self._feature_spec = feature_spec
        self._observation_spec = observation_spec

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

        self.feature_dim = feature_dim

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
        # if fuse_net is None:
        #     fuse_net = EncodingNetwork(
        #         name="inverse_net",
        #         input_tensor_spec=[feature_spec, feature_spec],
        #         fc_layer_params=hidden_size,
        #         last_layer_size=self._num_actions,
        #         last_kernel_initializer=tf.initializers.Zeros())

        self._fuse_net = fuse_net
        print("in----reward----------")
        print(fuse_net)
        print(self._fuse_net)
        # print("_fuse_net----------")
        # import pdb
        # pdb.set_trace()
        # print(self._fuse_net.variables)
        # print("--------reward estimation----------")
        # print(hidden_size)

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
        # print("=======in train step--------")
        # print(self._fuse_net.variables)

        feature, prev_action, reward_external = inputs

        prev_action = tf.math.floormod(prev_action, 3.142)  # mod operation

        # if self._encoding_net is not None:
        #     goal_feature, _, self_state_feature = self._encoding_net(feature)
        state_goal = feature['image']
        state_self = feature['state']

        # print("==========inside intrinsic module--------")
        # print(state_goal)
        # print(state_self)

        # print(state_goal)
        # print(state_self)

        prev_feature = state
        prev_action = self._encode_action(prev_action)

        # the reward estimation module should estimate next state and reward jointly (a complete forward model)

        # reward prediction loss based on forward_pred
        #self_pose = state_self  # all position based state now, no need to split

        # reward estimation based on one-step forward prediction
        reward_input_feature = tf.concat(
            [
                state_self,
                state_goal,  # train goal encoding
            ],
            axis=1)

        state_self_trans, _ = self._fuse_net(state_self)
        state_goal_trans, _ = self._fuse_net(state_goal)

        state_self_trans = tf.stop_gradient(state_self_trans)

        # reward_pred, _ = self._fuse_net(reward_input_feature)
        # mse based reward prediction
        # reward_pred = tf.exp(-tf.reduce_mean(
        #     tf.square(state_self - state_goal), axis=-1)) * 2 - 1

        diff = state_self - state_goal
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))
        step_penalty = -0.1
        reward_pred = tf.exp(-dist * 10) + step_penalty

        # print('----intrinsic ')
        # print(reward_pred)
        # print(dist)
        h_loss = tf.keras.losses.Huber()
        # reward_pred = 1 - 2 * tf.exp(
        #     -h_loss(state_self_trans, state_goal_trans) * 1)

        # reward_pred = tf.exp(
        #     -h_loss(state_self_trans, state_goal_trans) * 0.5)

        # pred_reward_mse = reward_pred

        pred_reward_mse = tf.reduce_mean(
            tf.square(state_self_trans - state_goal_trans), axis=-1)
        # reward_pred = -pred_reward_mse  # maximize reward, minimize difference through policy
        # # minimize loss difference

        pos_mask = tf.dtypes.cast(
            tf.math.greater(reward_external, tf.zeros_like(reward_external)),
            tf.float32)
        neg_mask = tf.dtypes.cast(
            tf.math.greater(-reward_external, tf.zeros_like(reward_external)),
            tf.float32)

        pos_total = tf.math.reduce_sum(pos_mask) + 1e-5
        neg_total = tf.math.reduce_sum(neg_mask) + 1e-5
        # print(pos_total)
        # print(neg_total)
        #reward_external_fuse = pos_mask - neg_mask

        pred_reward_mse = tf.multiply(pos_mask, pred_reward_mse) / pos_total

        # mse = 0.5 * tf.square(reward_pred - reward_external)
        # mse = 0.5 * tf.square(reward_pred - reward_external_fuse)
        # pos_reward_mse = tf.multiply(pos_mask, mse) / pos_total
        # neg_reward_mse = tf.multiply(neg_mask, mse) / neg_total

        #masked_reward = tf.multiply(scaled_mask, reward_external)

        # pred_reward_mse = 0.5 * (pos_reward_mse + neg_reward_mse)
        # reward_loss = 1e2 * pred_reward_mse
        #reward_loss = 1 * pred_reward_mse
        reward_loss = 0 * pred_reward_mse

        # pred_reward_mse = 0.5 * tf.square(
        #     reward_pred - reward_external) / total_sum  # reduce the last dim

        # reward_loss = 1e2 * tf.multiply(
        #     pred_reward_mse,
        #     tf.stop_gradient(scaled_mask))  # reduce the last dim

        # # # use a dummy reward loss in the second stage
        # reward_loss = 0. * tf.multiply(
        #     pred_reward_mse,
        #     tf.stop_gradient(scaled_mask))  # reduce the last dim

        #print(reward_pred)
        intrinsic_reward = ()
        if calc_intrinsic_reward:
            # intrinsic_reward = tf.stop_gradient(forward_loss)
            intrinsic_reward = tf.stop_gradient(reward_pred)
            # intrinsic_reward = self._reward_normalizer.normalize(
            #     intrinsic_reward)
            #print(intrinsic_reward)

        return AlgorithmStep(
            outputs=(),
            state=state_self,
            info=ICMInfo(
                reward=intrinsic_reward,
                loss=LossInfo(
                    loss=reward_loss, extra=dict(reward_loss=reward_loss))))

    def calc_loss(self, info: ICMInfo):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(scalar_loss=loss.loss, extra=loss.extra)
