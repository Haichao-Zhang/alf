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
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience

# a meta-algorithm that can take multiple algorithms as input

MultiAgentAlgorithmState = namedtuple("MetaState",
                                      ["teacher_state", "learner_state"])


@gin.configurable
class MultiAgentAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 algos,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 debug_summaries=False,
                 name="MultiAgentAlgorithm"):
        """Create an MultiAgentAlgorithm

        Args:
            algos: a list of algorithms. algorithm should come with an optimizer, if a module need to keep fixed, then the optimizer should be a dummy one with learning rate as 0

            optimizer: default optimizer for learning
            name (str): Name of this algorithm
            """

        # need actor_network
        # action_distribution_spec = actor_network.output_spec

        super(MultiAgentAlgorithm,
              self).__init__(action_spec=action_spec,
                             train_state_spec=train_state_spec,
                             action_distribution_spec=action_distribution_spec,
                             debug_summaries=debug_summaries,
                             name=name)
        # input_tensor_spec provides the observation dictionary
        self._action_spec = action_spec  # multi-agent action spec
        self._algos = algos

    def get_sliced_time_step(self, time_step: ActionTimeStep, idx):
        """Extract sliced time step information based on the specified index
        """
        return time_step._replace(observation=time_step.observation[idx],
                                  prev_action=time_step.prev_action[idx])

    def assemble_policy_step(self, policy_steps):
        assert len(policy_steps) > 1, "need more than one policy steps"
        actions = []
        states = []
        infos = []
        for i, ps in enumerate(policy_steps):
            actions.append(ps.action)
            states.append(ps.state)
            infos.append(ps.info)

        policy_step = policy_steps[0]
        policy_step._replace(action=actions, state=states, info=infos)

        return policy_step

    # rollout and train complete

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                with_experience=False):
        policy_steps = []
        for (i, algo) in enumerate(self._algos):
            time_step_sliced = self.get_sliced_time_step(time_step, i)
            state_sliced = state[i]
            policy_steps.append(
                algo.rollout(time_step_sliced, state_sliced, with_experience))

        return self.assemble_policy_step(policy_steps)

    def train_step(self, exp: Experience, state):
        time_step = ActionTimeStep(step_type=exp.step_type,
                                   reward=exp.reward,
                                   discount=exp.discount,
                                   observation=exp.observation,
                                   prev_action=exp.prev_action)
        return self.rollout(time_step, state, with_experience=True)

    # unnecessary
    # def train_complete(self,
    #                    tape: tf.GradientTape,
    #                    training_info,
    #                    valid_masks=None,
    #                    weight=1.0):
    #     time_step = ActionTimeStep(step_type=exp.step_type,
    #                                reward=exp.reward,
    #                                discount=exp.discount,
    #                                observation=exp.observation,
    #                                prev_action=exp.prev_action)
    #     return self.rollout(time_step, state, with_experience=True)