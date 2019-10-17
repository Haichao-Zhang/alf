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
from alf.algorithms.rl_algorithm import ActionTimeStep, TrainingInfo, RLAlgorithm, namedtuple
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience

# a meta-algorithm that can take multiple algorithms as input

MultiAgentState = namedtuple("MultiAgentState",
                             ["teacher_state", "learner_state"],
                             default_value=())

ActorCriticState = namedtuple("ActorCriticState", ["actor", "value"],
                              default_value=())

ActorCriticInfo = namedtuple("ActorCriticInfo", ["value"])


@gin.configurable
class MultiAgentAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 algo_ctors,
                 action_spec,
                 domain_names,
                 debug_summaries=False,
                 loss_class=ActorCriticLoss,
                 name="MultiAgentAlgorithm"):
        """Create an MultiAgentAlgorithm

        Args:
            algos: a list of algorithms. algorithm should come with an optimizer, if a module need to keep fixed, then the optimizer should be a dummy one with learning rate as 0

            optimizer: default optimizer for learning
            name (str): Name of this algorithm
            """

        # need actor_network
        # action_distribution_spec = actor_network.output_spec

        algos = [
            algo_ctor(debug_summaries=debug_summaries)
            for algo_ctor in algo_ctors
        ]

        def get_train_specs(algos):
            specs = OrderedDict()
            for i, algo in enumerate(algos):
                specs[domain_names[i]] = algo.train_state_spec
            return specs

        def get_predict_specs(algos):
            specs = OrderedDict()
            for i, algo in enumerate(algos):
                specs[domain_names[i]] = algo.predict_state_spec
            return specs

        def get_action_distribution_specs(algos):
            specs = OrderedDict()
            for i, algo in enumerate(algos):
                specs[domain_names[i]] = algo.action_distribution_spec
            return specs

        # def get_action_distribution_specs(algos):
        #     specs = []
        #     for algo in algos:
        #         specs.append(algo.action_distribution_spec)
        #     return specs

        super(MultiAgentAlgorithm, self).__init__(
            action_spec=action_spec,
            train_state_spec=get_train_specs(algos),
            predict_state_spec=get_predict_specs(algos),
            action_distribution_spec=get_action_distribution_specs(algos),
            debug_summaries=debug_summaries,
            name=name)
        # input_tensor_spec provides the observation dictionary
        self._action_spec = action_spec  # multi-agent action spec
        self._algo_ctors = algo_ctors
        self._algos = algos
        # self._algo0 = algos[0]
        # self._algo1 = algos[1]
        self._domain_names = domain_names
        self._debug_summaries = debug_summaries

    def get_sliced_time_step(self, time_step: ActionTimeStep, idx):
        """Extract sliced time step information based on the specified index
        """
        dn = self._domain_names[idx]
        return time_step._replace(observation=time_step.observation[dn],
                                  prev_action=time_step.prev_action[dn])

    def get_sliced_experience(self, exp, idx):
        """Extract sliced time step information based on the specified index
        Args:
            time_step (ActionTimeStep | Experience): time step
        Returns:
            ActionTimeStep | Experience: transformed time step
        """
        dn = self._domain_names[idx]
        return exp._replace(
            observation=exp.observation[dn],
            prev_action=exp.prev_action[dn],
            action=exp.action[dn],
            info=exp.info[dn],
            action_distribution=exp.action_distribution[dn],
            #state=exp.state[dn], # state is empty for non-RNN case
        )

    def assemble_experience(self, exps):
        # there could be new fields
        assert len(exps) > 1, "need more than one policy steps"
        observations = OrderedDict()
        prev_actions = OrderedDict()
        actions = OrderedDict()
        infos = OrderedDict()
        action_distributions = OrderedDict()
        #states = OrderedDict()

        for idx, ps in enumerate(exps):
            dn = self._domain_names[idx]
            observations[dn] = ps.observation
            prev_actions[dn] = ps.prev_action
            actions[dn] = ps.action
            action_distributions[dn] = ps.action_distribution
            infos[dn] = ps.info
            #states[dn] = ps.state
        # for i, ps in enumerate(policy_steps):
        #     actions.append(ps.action)
        #     states.append(ps.state)
        #     infos.append(ps.info)

        exp = exps[0]
        exp = exp._replace(
            observation=observations,
            prev_action=prev_actions,
            action=actions,
            action_distribution=action_distributions,
            info=infos,
            #state=states,
        )
        return exp

    def get_sliced_train_info(self, training_info: TrainingInfo, idx):
        """Extract sliced time step information based on the specified index
        """
        dn = self._domain_names[idx]
        return training_info._replace(
            action_distribution=training_info.action_distribution[dn],
            action=training_info.action[dn],
            collect_action_distribution=training_info.
            collect_action_distribution[dn],
            info=training_info.info[dn],
            collect_info=training_info.collect_info[dn],
        )

    def assemble_loss_info(self, loss_infos):
        assert len(loss_infos) > 1, "need more than one policy steps"
        losses = 0
        for idx, ps in enumerate(loss_infos):
            dn = self._domain_names[idx]
            losses += ps.loss

        loss_infos = loss_infos[0]
        loss_infos = loss_infos._replace(loss=losses)
        return loss_infos

    def get_sliced_state(self, state, idx):
        dn = self._domain_names[idx]
        return state[dn]

    def assemble_policy_step(self, policy_steps):
        assert len(policy_steps) > 1, "need more than one policy steps"
        actions = OrderedDict()
        states = OrderedDict()
        infos = OrderedDict()
        for idx, ps in enumerate(policy_steps):
            dn = self._domain_names[idx]
            actions[dn] = ps.action
            states[dn] = ps.state
            infos[dn] = ps.info
        # for i, ps in enumerate(policy_steps):
        #     actions.append(ps.action)
        #     states.append(ps.state)
        #     infos.append(ps.info)

        policy_step = policy_steps[0]
        policy_step = policy_step._replace(action=actions,
                                           state=states,
                                           info=infos)
        return policy_step

    @property
    def trainable_variables(self):
        return sum([var_set for _, var_set in self._get_opt_and_var_sets()],
                   [])

    # def transform_timestep(self, time_step):
    #     policy_steps = []
    #     for (i, algo) in enumerate(self._algos):
    #         time_step_sliced = self.get_sliced_time_step(time_step, i)

    def preprocess_experience(self, exp: Experience):
        exps = []
        for (i, algo) in enumerate(self._algos):
            exp_sliced = self.get_sliced_experience(exp, i)
            exps.append(algo.preprocess_experience(exp_sliced))
        return self.assemble_experience(exps)

    # rollout and train complete

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                with_experience=False):
        policy_steps = []
        for (i, algo) in enumerate(self._algos):
            time_step_sliced = self.get_sliced_time_step(time_step, i)
            state_sliced = self.get_sliced_state(state, i)
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

    # this is used in train_complete
    def calc_loss(self, training_info):
        """Calculate loss."""
        loss_infos = []
        for (i, algo) in enumerate(self._algos):
            print("================loss info-------")
            print(self.get_sliced_train_info(training_info, i))

            loss_info_sliced = algo.calc_loss(
                self.get_sliced_train_info(training_info, i))

            loss_infos.append(loss_info_sliced)

        print("================loss info-------")
        print(self.assemble_loss_info(loss_infos))
        return self.assemble_loss_info(loss_infos)

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