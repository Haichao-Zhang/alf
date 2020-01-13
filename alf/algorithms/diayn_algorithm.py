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

DIAYNInfo = namedtuple("DIAYNInfo", ["reward", "loss"])


@gin.configurable
class DIAYNAlgorithm(Algorithm):
    """Diversity is All You Need Module

    This module generate the intrinsic reward based on predition error of
    observation.

    See Eysenbach et al "Diversity is All You Need: Learning Diverse Skills without a Reward Function"
    """

    def __init__(self,
                 num_of_skills,
                 feature_spec,
                 hidden_size=256,
                 reward_adapt_speed=8.0,
                 encoding_net: Network = None,
                 discriminator_net: Network = None,
                 name="DIAYNAlgorithm"):
        """Create a DIAYNAlgorithm.

        Args:
            num_of_skills: number of skills (goals)
            hidden_size (int|tuple): size of hidden layer(s)
            reward_adapt_speed (float): how fast to adapt the reward normalizer.
                roughly speaking, the statistics for the normalization is
                calculated mostly based on the most recent T/speed samples,
                where T is the total number of samples.
            encoding_net (Network): network for encoding observation into a
                latent feature specified by feature_spec. Its input is same as
                the input of this algorithm.
            discriminator_net (Network): network for predicting the skill labels based on the observation.
        """
        super(DIAYNAlgorithm, self).__init__(
            train_state_spec=feature_spec, name=name)

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(flat_feature_spec
                   ) == 1, "DIAYNAlgorithm doesn't support nested feature_spec"

        self._num_skills = num_of_skills
        self._skill_spec = tf.TensorSpec(
            shape=(self._num_skills, ), dtype=tf.int32)

        # # should be able to handle both discrete and continious case
        # if tensor_spec.is_discrete(skill_spec):
        #     self._num_skills = skill_spec.maximum - skill_spec.minimum + 1
        # else:
        #     self._num_skills = skill_spec.shape[-1]

        #feature_dim = flat_feature_spec[0].shape[-1]

        self._encoding_net = encoding_net

        if isinstance(hidden_size, int):
            hidden_size = (hidden_size, )

        if discriminator_net is None:
            discriminator_net = EncodingNetwork(
                name="discriminator_net",
                input_tensor_spec=feature_spec,
                fc_layer_params=hidden_size,
                last_layer_size=self._num_skills,
                last_kernel_initializer=tf.initializers.Zeros())

        self._discriminator_net = discriminator_net

    def _encode_skill(self, skill):
        if tensor_spec.is_discrete(self._skill_spec):
            return tf.one_hot(indices=skill, depth=self._num_skills)
        else:
            return skill

    def train_step(self, inputs, state, calc_intrinsic_reward=True):
        """
        Args:
            inputs (tuple): observation and previous action
            state (Tensor): state for DIAYN (previous observation)
            calc_intrinsic_reward (bool): if False, only return the losses
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: observation
                info (DIAYNInfo):
        """
        observations_aug, _ = inputs
        feature = observations_aug['obs']
        skill = observations_aug['goal']

        # if self._encoding_net is not None:
        #     feature, _ = self._encoding_net(feature)
        if self._encoding_net is not None:
            feature, _, _ = self._encoding_net(observations_aug)

        prev_feature = state
        skill = tf.cast(skill, tf.int32)
        skill = self._encode_skill(skill)

        skill_pred, _ = self._discriminator_net(inputs=feature)

        if tensor_spec.is_discrete(self._skill_spec):
            skill_discriminate_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=skill, logits=skill_pred)
        else:
            skill_discriminate_loss = 0.5 * tf.reduce_mean(
                tf.square(skill - skill_pred), axis=-1)

        intrinsic_reward = ()
        if calc_intrinsic_reward:
            # use negative cross-entropy as reward
            # neglect the constant neg-prior term for now
            intrinsic_reward = tf.stop_gradient(-skill_discriminate_loss)

        return AlgorithmStep(
            outputs=(),
            state=feature,
            info=DIAYNInfo(
                reward=intrinsic_reward,
                loss=LossInfo(
                    loss=skill_discriminate_loss,
                    extra=dict(
                        skill_discriminate_loss=skill_discriminate_loss))))

    def calc_loss(self, info: DIAYNInfo):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(scalar_loss=loss.loss, extra=loss.extra)
