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
import numpy as np
import os

#from tf_agents.agents.tf_agent import LossInfo  # this LossInfo is not correct
# use the one from common.py

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
from alf.algorithms.rl_algorithm import ActionTimeStep, TrainingInfo, LossInfo, RLAlgorithm, namedtuple
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience

# a meta-algorithm that can take multiple algorithms as input

MultiAgentState = namedtuple(
    "MultiAgentState", ["teacher_state", "learner_state", "icm"],
    default_value=())

AgentState = namedtuple("AgentState", ["rl", "icm", "value"], default_value=())

AgentInfo = namedtuple("AgentInfo", ["rl", "icm", "value"], default_value=())

AgentLossInfo = namedtuple("AgentLossInfo", ["rl", "icm"], default_value=())


@gin.configurable
class MultiAgentAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 algo_ctors,
                 action_spec,
                 domain_names,
                 teacher_training_phase,
                 intrinsic_curiosity_module=None,
                 intrinsic_reward_coef=1.0,
                 extrinsic_reward_coef=1.0,
                 debug_summaries=False,
                 loss_class=ActorCriticLoss,
                 reward_shaping_fn: Callable = None,
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
            # no need to consider intrinsic module in this function
            specs = OrderedDict()
            for i, algo in enumerate(algos):
                if ((not teacher_training_phase and i == 0) or
                    (teacher_training_phase
                     and i == 1)) and intrinsic_curiosity_module is not None:
                    train_state_spec = AgentState(rl=algo.train_state_spec)
                    train_state_spec = train_state_spec._replace(
                        icm=intrinsic_curiosity_module.train_state_spec)
                else:
                    train_state_spec = algo.train_state_spec

                specs[domain_names[i]] = train_state_spec
            return specs

        def get_predict_specs(algos):
            specs = OrderedDict()
            for i, algo in enumerate(algos):
                if ((not teacher_training_phase and i == 0) or
                    (teacher_training_phase
                     and i == 1)) and intrinsic_curiosity_module is not None:
                    predict_state_spec = AgentState(rl=algo.predict_state_spec)
                    predict_state_spec = predict_state_spec._replace(
                        icm=intrinsic_curiosity_module.predict_state_spec)
                else:
                    predict_state_spec = algo.predict_state_spec

                specs[domain_names[i]] = predict_state_spec
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
            reward_shaping_fn=reward_shaping_fn,
            observation_transformer=observation_transformer,
            name=name)
        # input_tensor_spec provides the observation dictionary
        self._action_spec = action_spec  # multi-agent action spec
        self._algo_ctors = algo_ctors
        self._algos = algos
        self._domain_names = domain_names
        self._debug_summaries = debug_summaries
        self._teacher_training_phase = teacher_training_phase

        self._intrinsic_reward_coef = intrinsic_reward_coef
        self._extrinsic_reward_coef = extrinsic_reward_coef
        self._icm = intrinsic_curiosity_module

    def load_part_model(self, root_dir):
        # load pre-trained variables
        if self._icm is not None:
            self._icm_checkpoint = tf.train.Checkpoint(icm=self._icm)
            ckpt_dir = os.path.join(root_dir, 'icm')
            latest = tf.train.latest_checkpoint(ckpt_dir)
            if latest is not None:
                #print(self._icm._fuse_net.summary())
                #print(self._icm._fuse_net.variables)
                print("-------load ICM model-----")
                print(latest)
                self._icm_checkpoint.restore(latest)
                #print(self._icm._fuse_net.summary())
                print(self._icm._fuse_net.variables)
            else:
                print("-------No ICM checkpoint-----")
                #print(self._icm._fuse_net.summary())
                #print(self._icm._fuse_net.variables)

    def save_part_model(self, root_dir):
        # ckpt_dir = os.path.join(root_dir, 'icm')
        # self._icm_checkpoint.save(ckpt_dir + '/ck')
        return

    def get_sliced_data(self, data, domain_name):
        """Extract sliced time step information based on the specified index
        Args:
            data is in the form of named tuple
            or dictionary
        """
        #assert type(data) is namedtuple, "input data should instance of namedtuple"
        fields = data._fields
        for fd in fields:
            field_data = getattr(data, fd)
            if type(field_data) is not OrderedDict:
                continue
            if domain_name in field_data.keys():
                data = data._replace(**{fd: field_data[domain_name]})
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
            #reward[dn] = ps.reward
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
        print("=======optimizer------")
        print(optimizer_and_module_sets)
        #for opt, module_set in optimizer_and_module_sets:
        if self._teacher_training_phase:
            return optimizer_and_module_sets[1:]
        else:
            return optimizer_and_module_sets[0:1]

    # @property
    # def trainable_variables(self):
    #     opt_set, var_set = self._get_opt_and_var_sets()
    #     res = sum([var_set for _, var_set in self._get_opt_and_var_sets()], [])
    #     print("=======================")
    #     print(res)
    #     return res
    def calc_training_reward(self, external_reward, info, with_icm_flag):
        """Calculate the reward actually used for training.

        The training_reward includes both intrinsic reward (if there's any) and
        the external reward.
        Args:
            external_reward (Tensor): reward from environment
            info (ActorCriticInfo): (batched) policy_step.info from train_step()
        Returns:
            reward used for training.
        """
        # record shaped extrinsic rewards actually used for training
        self.add_reward_summary("reward/extrinsic", external_reward)
        reward = self._extrinsic_reward_coef * external_reward

        if with_icm_flag:
            self.add_reward_summary("reward/icm", info.icm.reward)
            reward += self._intrinsic_reward_coef * info.icm.reward

        if id(reward) != id(external_reward):
            self.add_reward_summary("reward/overall", reward)

        return reward

    def preprocess_experience(self, exp: Experience):
        exps = []
        for (i, algo) in enumerate(self._algos):
            exp_sliced = self.get_sliced_experience(exp, i)
            if ((not self._teacher_training_phase and i == 0) or
                (self._teacher_training_phase
                 and i == 1)) and self._icm is not None:
                # print('----exp')
                # print(exp_sliced.info.icm.reward)
                print('---pre-process-------')

                reward = self.calc_training_reward(
                    exp_sliced.reward, exp_sliced.info, True)  #info.rl
                exp_sliced = exp_sliced._replace(reward=reward)

                new_info = exp_sliced.info._replace(
                    value=exp_sliced.info.rl.value)
                exp_sliced = exp_sliced._replace(info=new_info)

                exp_sliced = algo.preprocess_experience(exp_sliced)

                exps.append(exp_sliced)
            else:
                # print("--preprocessing 2")
                # reward = self.calc_training_reward(exp_sliced.reward,
                #                                    exp_sliced.info,
                #                                    False)  #info.rl
                # avoid calling the calc_training_reward (for avoiding recording into tb)
                reward = exp_sliced.reward
                exps.append(
                    algo.preprocess_experience(
                        exp_sliced._replace(reward=reward)))
        # reward is the same
        return self.assemble_experience(exps)._replace(reward=reward)

    # def transform_timestep(self, exp: Experience):
    #     exps = []
    #     for (i, algo) in enumerate(self._algos):
    #         exp_sliced = self.get_sliced_experience(exp, i)
    #         exps.append(algo.transform_timestep(exp_sliced))
    #     return self.assemble_experience(exps)

    # rollout and train complete
    def predict(self, time_step: ActionTimeStep, state):
        """Predict for one step."""
        policy_steps = []
        for (i, algo) in enumerate(self._algos):
            # time_step_sliced = self.get_sliced_time_step(time_step, i)
            #state_sliced = self.get_sliced_state(state, i)
            time_step_sliced = self.get_sliced_data(time_step,
                                                    self._domain_names[i])
            state_sliced = self.get_sliced_state(state, i)
            rl_step = algo.predict(time_step_sliced, state_sliced)

            #if self._icm is not None:
            if ((not self._teacher_training_phase and i == 0) or
                (self._teacher_training_phase
                 and i == 1)) and self._icm is not None:
                new_state = AgentState()
                new_state = new_state._replace(rl=rl_step.state)
                rl_step = PolicyStep(
                    action=rl_step.action, state=new_state, info=())
                policy_steps.append(rl_step)
            else:
                policy_steps.append(rl_step)

        return self.assemble_policy_step(policy_steps)

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                with_experience=False):
        policy_steps = []
        for (i, algo) in enumerate(self._algos):
            new_state = AgentState()
            info = AgentInfo()
            #observation = self._encode(time_step)

            time_step_sliced = self.get_sliced_time_step(time_step, i)
            state_sliced = self.get_sliced_state(state, i)
            # seperate algo.rollout
            # todo: only add icm to one of the algorithm
            if ((not self._teacher_training_phase and i == 0) or
                (self._teacher_training_phase
                 and i == 1)) and self._icm is not None:
                #print("osbervation") # ['image', 'state']
                # print(time_step_sliced.observation)
                icm_step = self._icm.train_step(
                    (time_step_sliced.observation,
                     time_step_sliced.prev_action, time_step_sliced.reward),
                    state=state_sliced.icm,
                    calc_intrinsic_reward=not with_experience)
                info = info._replace(icm=icm_step.info)
                new_state = new_state._replace(icm=icm_step.state)

                rl_step = algo.rollout(time_step_sliced, state_sliced.rl,
                                       with_experience)

                new_state = new_state._replace(rl=rl_step.state)
                info = info._replace(rl=rl_step.info)
                # clipped_action = tf.clip_by_value(rl_step.action,
                #                                   clip_value_min=-3.142,
                #                                   clip_value_max=3.142)
                # print("----action")
                # print(clipped_action)
                # create new policy step
                new_policy_step = PolicyStep(
                    action=rl_step.action, state=new_state, info=info)
                # missing the reward here
                policy_steps.append(new_policy_step)
            else:

                rl_step = algo.rollout(time_step_sliced, state_sliced,
                                       with_experience)
                policy_steps.append(rl_step)

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

        def _add(x, y):
            if not isinstance(y, tf.Tensor):
                return x
            elif not isinstance(x, tf.Tensor):
                return y
            else:
                return x + y

        def _update_loss(loss_info, training_info, name, algorithm):
            if algorithm is None:
                return loss_info
            new_loss_info = algorithm.calc_loss(
                getattr(training_info.info, name))
            return LossInfo(
                loss=_add(loss_info.loss, new_loss_info.loss),
                scalar_loss=_add(loss_info.scalar_loss,
                                 new_loss_info.scalar_loss),
                extra=loss_info.extra._replace(**{name: new_loss_info.extra}))

        loss_infos = []
        for (i, algo) in enumerate(self._algos):
            #print(training_info)
            if ((not self._teacher_training_phase and i == 0) or
                (self._teacher_training_phase
                 and i == 1)) and self._icm is not None:

                # RL loss plus reward prediction loss
                sliced_info = self.get_sliced_train_info(training_info, i)

                info_rl = sliced_info.info.rl
                # print('---loss=====')
                # print(sliced_info.info.icm)
                # import pdb
                # pdb.set_trace()
                if sliced_info.collect_info == ():
                    # import pdb
                    # pdb.set_trace()
                    sliced_info = sliced_info._replace(
                        reward=self.calc_training_reward(
                            sliced_info.reward, sliced_info.info, True))

                loss_info_sliced_rl = algo.calc_loss(
                    sliced_info._replace(info=sliced_info.info.rl))
                # ToDo: reward prediction loss
                # import pdb
                # pdb.set_trace()
                loss_info_sliced = loss_info_sliced_rl._replace(
                    extra=AgentLossInfo(rl=loss_info_sliced_rl.extra))
                loss_info_sliced = _update_loss(loss_info_sliced, sliced_info,
                                                'icm', self._icm)
            else:
                loss_info_sliced = algo.calc_loss(
                    self.get_sliced_train_info(training_info, i))

            loss_infos.append(loss_info_sliced)
        # return self.assemble_loss_info(loss_infos)
        if self._teacher_training_phase:
            return loss_infos[1]  # return teacher loss in this stage
        else:
            return loss_infos[0]  # return learner loss in this stage

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
