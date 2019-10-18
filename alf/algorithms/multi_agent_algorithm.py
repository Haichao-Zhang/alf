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

MultiAgentState = namedtuple(
    "MultiAgentState", ["teacher_state", "learner_state"], default_value=())


@gin.configurable
class MultiAgentAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 algo_ctors,
                 action_spec,
                 domain_names,
                 debug_summaries=False,
                 loss_class=ActorCriticLoss,
                 observation_transformer: Callable = None,
                 name="MultiAgentAlgorithm"):
        """Create an MultiAgentAlgorithm

        Args:
            algo_ctors: a list of constructors for algorithms. An algorithm should come with an optimizer.
            domain_names: a list containing the names of all agents; the name of the agent should be the same as the one used in the nested observation/control
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

        super(MultiAgentAlgorithm, self).__init__(
            action_spec=action_spec,
            train_state_spec=get_train_specs(algos),
            predict_state_spec=get_predict_specs(algos),
            action_distribution_spec=get_action_distribution_specs(algos),
            debug_summaries=debug_summaries,
            optimizer=[],
            trainable_module_sets=[],
            observation_transformer=observation_transformer,
            name=name)
        # input_tensor_spec provides the observation dictionary
        self._action_spec = action_spec  # multi-agent action spec
        self._algo_ctors = algo_ctors
        self._algos = algos
        self._domain_names = domain_names
        self._debug_summaries = debug_summaries

    def get_sliced_data(self, data, domain_name):
        """Extract sliced time step information based on the specified index
        Args:
            data is in the form of named tuple
        """
        assert type(
            data) is namedtuple, "input data should instance of namedtuple"
        fields = data._fields
        for fd in fields:
            if domain_name in data[fd].keys():
                data = data._replace(fd=data[fd][domain])
        return data

    def get_sliced_time_step(self, time_step: ActionTimeStep, idx):
        """Extract sliced time step information based on the specified index
        """
        dn = self._domain_names[idx]
        return time_step._replace(
            observation=time_step.observation[dn],
            prev_action=time_step.prev_action[dn])

    def get_sliced_experience(self, exp, idx):
        """Extract sliced time step information based on the specified index
        Args:
            exp: Experience
        Returns:
            sliced_exp: sliced Experience
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
        policy_step = policy_step._replace(
            action=actions, state=states, info=infos)
        return policy_step

    def get_optimizer_and_module_sets(self):
        """
        Return specified optimizer and module sets
        """
        optimizer_and_module_sets = super().get_optimizer_and_module_sets()
        #for opt, module_set in optimizer_and_module_sets:
        return optimizer_and_module_sets[1:]

    # @property
    # def trainable_variables(self):
    #     opt_set, var_set = self._get_opt_and_var_sets()
    #     res = sum([var_set for _, var_set in self._get_opt_and_var_sets()], [])
    #     print("=======================")
    #     print(res)
    #     return res

    def preprocess_experience(self, exp: Experience):
        exps = []
        for (i, algo) in enumerate(self._algos):
            exp_sliced = self.get_sliced_experience(exp, i)
            exps.append(algo.preprocess_experience(exp_sliced))
        return self.assemble_experience(exps)

    # def transform_timestep(self, exp: Experience):
    #     exps = []
    #     for (i, algo) in enumerate(self._algos):
    #         exp_sliced = self.get_sliced_experience(exp, i)
    #         exps.append(algo.transform_timestep(exp_sliced))
    #     return self.assemble_experience(exps)

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
        time_step = ActionTimeStep(
            step_type=exp.step_type,
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
            loss_info_sliced = algo.calc_loss(
                self.get_sliced_train_info(training_info, i))

            loss_infos.append(loss_info_sliced)
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
