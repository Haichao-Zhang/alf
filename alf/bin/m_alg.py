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

import gin.tf.external_configurables
from alf.utils import common
import alf.utils.external_configurables
from alf.trainers import policy_trainer

from alf.trainers import on_policy_trainer
from alf.trainers import off_policy_trainer
from alf.algorithms import agent
from alf.algorithms import multi_agent_algorithm
from alf.algorithms import reward_estimation
from alf.algorithms import ppo_algorithm

from alf.utils import multi_modal_actor_distribution_network
from tf_agents.networks.encoding_network import EncodingNetwork

from alf.environments.utils import create_environment
from alf.utils import common


@gin.configurable
def get_m_algo():
    with gin.config_scope('reward'):
        feature_net = EncodingNetwork()
    print("----feature net==============")
    print(feature_net)

    print('featurenet before reward estimator')
    print(feature_net)
    # as separate fixed network
    with gin.config_scope('reward'):
        feature_net2 = EncodingNetwork()

    reward_estimator = reward_estimation.RewardAlgorithmState(
        fuse_net=feature_net)

    # print("para value after creation-----=====================-----")
    # print(feature_net.variables)
    with gin.config_scope('learner'):
        learner_network = multi_modal_actor_distribution_network.MultiModalActorDistributionNetworkMapping(
            feature_mapping=feature_net)
        learner_ppo_algo = ppo_algorithm.PPOAlgorithm(
            actor_network=learner_network)
    with gin.config_scope('teacher'):
        teacher_ppo_algo = ppo_algorithm.PPOAlgorithm()

    # print("------reward estimator==================")
    # print(reward_estimator._fuse_net.variables)
    # print("------ppo*******************************")
    # print(learner_ppo_algo._actor_network._feature_mapping.variables)

    m_alg = multi_agent_algorithm.MultiAgentAlgorithm(
        intrinsic_curiosity_module=reward_estimator,
        algos=[learner_ppo_algo, teacher_ppo_algo],
        debug_summaries=True)

    return m_alg
