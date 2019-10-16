from collections import namedtuple, OrderedDict
import functools
from typing import Callable

import gin.tf

import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.trajectories.policy_step import PolicyStep

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.entropy_target_algorithm import EntropyTargetAlgorithm
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.rl_algorithm import ActionTimeStep, TrainingInfo, RLAlgorithm
from alf.algorithms.algorithm import Algorithm

# a meta-algorithm that can take multiple algorithms as input

MultiAgentAlgorithmState = namedtuple("MetaState",
                                      ["teacher_state", "learner_state"])


@gin.configurable
class MultiAgentAlgorithm(RLAlgorithm):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 algos,
                 debug_summaries=False,
                 name="MultiAgentAlgorithm"):
        """Create an MultiAgentAlgorithm

        Args:
            algos: a list of algorithms. algorithm should come with an optimizer, if a module need to keep fixed, then the optimizer should be a dummy one with learning rate as 0

            optimizer: default optimizer for learning
            name (str): Name of this algorithm
            """

        # need actor_network
        action_distribution_spec = actor_network.output_spec

        super(MultiAgentAlgorithm,
              self).__init__(action_spec=action_spec,
                             action_distribution_spec=action_distribution_spec,
                             debug_summaries=debug_summaries,
                             name=name)
        # input_tensor_spec provides the observation dictionary
        self._action_spec = action_spec  # multi-agent action spec
        self._algos = algos
        # later get from environments and specified with gin
        self._observation_domain = {
            'observation_learner', 'observation_teacher'
        }
        self._action_domain = {'control_learner', 'control_teacher'}

    def slice_with_name(self, ActionTimeStep, name):
        ActionTimeStep

    def assemble_control(self, action_distribution,
                         action_distribution_teacher):
        joint_action_distribution = OrderedDict(
            control_learner=action_distribution,
            control_teacher=action_distribution_teacher)
        return joint_action_distribution

    def assemble_control(self, action_distribution,
                         action_distribution_teacher):
        joint_action_distribution = OrderedDict(
            control_learner=action_distribution,
            control_teacher=action_distribution_teacher)
        return joint_action_distribution

    # rollout and train complete

    def rollout(self, time_step: ActionTimeStep, state=None):
        policy_steps = []
        for (i, algo) in self._algos):
            time_step_sliced = ActionTimeStep(
                step_type=time_step.step_type,
                reward=time_step.reward,  # multi-reward TBD
                discount=time_step.discount,
                observation=time_step.observation[self._observation_domain[i]],
                prev_action=time_step.prev_action[self._action_domain[i]])
            state_sliced = state[self._observation_domain[i]]
            policy_steps.append(algo.rollout(time_step_sliced,
                                             state_sliced))  # state TBD

        return PolicyStep(action=joint_action_distribution,
                          state=state,
                          info=info)

    def calc_training_reward(self, external_reward, info: ActorCriticInfo):
        """Calculate the reward actually used for training.

        The training_reward includes both intrinsic reward (if there's any) and
        the external reward.
        Args:
            external_reward (Tensor): reward from environment
            info (ActorCriticInfo): (batched) policy_step.info from train_step()
        Returns:
            reward used for training.
        """
        if self._icm is not None:
            return (self._extrinsic_reward_coef * external_reward +
                    self._intrinsic_reward_coef * info.icm_reward)
        else:
            return external_reward

    # over
    def calc_loss(self, training_info):
        if self._icm is not None:
            self.add_reward_summary("reward/intrinsic",
                                    training_info.info.icm_reward)

            training_info = training_info._replace(
                reward=self.calc_training_reward(training_info.reward,
                                                 training_info.info))

            self.add_reward_summary("reward/overall", training_info.reward)

        # this part should be adapted for multi-agent case
        ac_loss = self._loss(training_info, training_info.info.value)
        loss = ac_loss.loss
        extra = ActorCriticAlgorithmLossInfo(ac=ac_loss.extra,
                                             icm=(),
                                             entropy_target=())

        if self._icm is not None:
            icm_loss = self._icm.calc_loss(training_info.info.icm_info)
            loss += icm_loss.loss
            extra = extra._replace(icm=icm_loss.extra)

        if self._entropy_target_algorithm:
            et_loss = self._entropy_target_algorithm.calc_loss(
                training_info.info.entropy_target_info)
            loss += et_loss.loss
            extra = extra._replace(entropy_target=et_loss.extra)

        return LossInfo(loss=loss, extra=extra)