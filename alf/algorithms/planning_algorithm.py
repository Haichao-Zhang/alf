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

from collections import namedtuple
import gin

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 TimeStep, TrainingInfo)
from alf.nest import nest
from alf.optimizers.random import RandomOptimizer, QOptimizer

from alf.utils import vis_utils, beam_search, sampling_utils, tensor_utils

from alf.utils.summary_utils import safe_mean_hist_summary

PlannerState = namedtuple("PlannerState", ["policy"], default_value=())
# PlannerInfo = namedtuple("PlannerInfo", ["policy", "loss"])  # can add loss
PlannerInfo = namedtuple("PlannerInfo", ["policy"])  # TODO: add loss
PlannerLossInfo = namedtuple('PlannerLossInfo', ["policy"])


@gin.configurable
class PlanAlgorithm(OffPolicyAlgorithm):
    """Planning Module

    This module plans for actions based on initial observation
    and specified reward and dynamics functions
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 train_state_spec,
                 planning_horizon=25,
                 upper_bound=None,
                 lower_bound=None,
                 name="PlanningAlgorithm"):
        """Create a PlanningAlgorithm.

        Args:
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
        """
        super().__init__(
            feature_spec,
            action_spec,
            train_state_spec=train_state_spec,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        assert action_spec.is_continuous, "only support \
                                                    continious control"

        self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._feature_spec = feature_spec
        self._planning_horizon = planning_horizon
        self._upper_bound = action_spec.maximum if upper_bound is None \
                                                else upper_bound
        self._lower_bound = action_spec.minimum if lower_bound is None \
                                                else lower_bound

        self._reward_func = None
        self._dynamics_func = None
        self._step_eval_func = None  # per step evaluation function

    def train_step(self, time_step: TimeStep, state):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                output: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        pass

    def set_step_eval_func(self, eval_func):
        """Set per-time-step reward function used for planning
        Args:
            eval_func (Callable): the evaluation function to be used for
                generating action proposals.
            eval_func takes (obs, action) as input
        """
        self._step_eval_func = eval_func

    def set_reward_func(self, reward_func):
        """Set per-time-step reward function used for planning
        Args:
            reward_func (Callable): the reward function to be used for planning.
            reward_func takes (obs, action) as input
        """
        self._reward_func = reward_func

    def set_dynamics_func(self, dynamics_func):
        """Set the dynamics function for planning
        Args:
            dynamics_func (Callable): reward function to be used for planning.
            dynamics_func takes (time_step, state) as input and returns
            next_time_step (TimeStep) and the next_state
        """
        self._dynamics_func = dynamics_func

    def generate_plan(self, time_step: TimeStep, state, epsilon_greedy):
        """Compute the plan based on the provided observation and action
        Args:
            time_step (TimeStep): input data for next step prediction
            state: input state next step prediction
        Returns:
            action: planned action for the given inputs
            state: mbrl state
        """
        pass

    # # never used
    # def calc_loss(self, info):
    #     loss = nest.map_structure(torch.mean, info.loss)
    #     return LossInfo(
    #         loss=info.loss, scalar_loss=loss.loss, extra=loss.extra)


@gin.configurable
class RandomShootingAlgorithm(PlanAlgorithm):
    """Random Shooting-based planning method.
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 population_size,
                 planning_horizon,
                 upper_bound=None,
                 lower_bound=None,
                 hidden_size=256,
                 name="RandomShootingAlgorithm"):
        """Create a RandomShootingAlgorithm.

        Args:
            population_size (int): the size of polulation for random shooting
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
            hidden_size (int|tuple): size of hidden layer(s)
        """
        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            train_state_spec=(),
            planning_horizon=planning_horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, ("RandomShootingAlgorithm doesn't "
                                            "support nested action_spec")

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(flat_feature_spec) == 1, ("RandomShootingAlgorithm doesn't "
                                             "support nested feature_spec")

        self._population_size = population_size
        solution_size = self._planning_horizon * self._num_actions
        self._plan_optimizer = RandomOptimizer(
            solution_size,
            self._population_size,
            upper_bound=action_spec.maximum,
            lower_bound=action_spec.minimum)

    def train_step(self, time_step: TimeStep, state):
        """
        Args:
            time_step (TimeStep): input data for planning
            state: state for planning (previous observation)
        Returns:
            TrainStep:
                output: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        return AlgStep(output=(), state=(), info=())

    def generate_plan(self, time_step: TimeStep, state, epsilon_greedy):
        assert self._reward_func is not None, ("specify reward function "
                                               "before planning")

        assert self._dynamics_func is not None, ("specify dynamics function "
                                                 "before planning")

        self._plan_optimizer.set_cost(self._calc_cost_for_action_sequence)
        opt_action = self._plan_optimizer.obtain_solution(time_step, state)
        # [B, T]
        action = opt_action[:, 0]
        action = torch.reshape(action, [time_step.observation.shape[0], -1])
        return action, state

    def _expand_to_population(self, data):
        """Expand the input tensor to a population of replications
        Args:
            data (Tensor): input data with shape [batch_size, ...]
        Returns:
            data_population (Tensor) with shape
                                    [batch_size * self._population_size, ...].
            For example data tensor [[a, b], [c, d]] and a population_size of 2,
            we have the following data_population tensor as output
                                    [[a, b], [a, b], [c, d], [c, d]]
        """
        data_population = torch.repeat_interleave(
            data, self._population_size, dim=0)
        return data_population

    def _calc_cost_for_action_sequence(self, time_step: TimeStep, state,
                                       ac_seqs):
        """
        Args:
            time_step (TimeStep): input data for next step prediction
            state (MbrlState): input state for next step prediction
            ac_seqs: action_sequence (Tensor) of shape [batch_size,
                    population_size, solution_dim]), where
                    solution_dim = planning_horizon * num_actions
        Returns:
            cost (tf.Tensor) with shape [batch_size, population_size]
        """
        obs = time_step.observation
        batch_size = obs.shape[0]

        ac_seqs = torch.reshape(
            ac_seqs,
            [batch_size, self._population_size, self._planning_horizon, -1])

        ac_seqs = ac_seqs.permute(2, 0, 1, 3)
        ac_seqs = torch.reshape(
            ac_seqs, (self._planning_horizon, -1, self._num_actions))

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        state = nest.map_structure(self._expand_to_population, state)

        obs = init_obs
        cost = 0
        for i in range(ac_seqs.shape[0]):
            action = ac_seqs[i]
            time_step = time_step._replace(prev_action=action)
            time_step, state = self._dynamics_func(time_step, state)
            next_obs = time_step.observation
            # Note: currently using (next_obs, action), might need to
            # consider (obs, action) in order to be more compatible
            # with the conventional definition of reward function
            reward_step = self._reward_func(next_obs, action)
            cost = cost - reward_step
            obs = next_obs

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])

        # reshape and permute back
        ac_seqs = torch.reshape(ac_seqs, [
            self._planning_horizon, batch_size, self._population_size,
            self._num_actions
        ]).permute(1, 2, 0, 3)
        return cost, ac_seqs

    def after_update(self, training_info):
        pass


@gin.configurable
class QShootingAlgorithm(PlanAlgorithm):
    """Q-value guided planning method.
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 population_size,
                 planning_horizon,
                 policy_module=None,
                 upper_bound=None,
                 lower_bound=None,
                 hidden_size=256,
                 repeat_times=10,
                 name="QShootingAlgorithm"):
        """Create a QShootingAlgorithm.
        Args:
            population_size (int): the size of polulation for Q shooting
            planning_horizon (int): planning horizon in terms of time steps
            policy_module (Algorithm): if None, reduces to MPC
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
            hidden_size (int|tuple): size of hidden layer(s)
        """
        train_state_spec = PlannerState(policy=policy_module.train_state_spec)
        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            planning_horizon=planning_horizon,
            train_state_spec=train_state_spec,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, ("QShootingAlgorithm doesn't "
                                            "support nested action_spec")

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(flat_feature_spec) == 1, ("QShootingAlgorithm doesn't "
                                             "support nested feature_spec")

        self._population_size = population_size
        solution_size = self._planning_horizon * self._num_actions
        self._plan_optimizer = QOptimizer(
            solution_size,
            self._population_size,
            upper_bound=action_spec.maximum,
            lower_bound=action_spec.minimum)

        self._policy_module = policy_module
        # # setup optimizers
        # self._policy_module._setup_optimizers()
        self._discount = 0.9
        self._repeat_times = repeat_times

        self._has_been_trained = False

    def train_step(self, exp: Experience, state):
        """ Performing Q (actor and critic) training
        Args:
            exp (Experience): input data for planning
            state: state for planning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        # set dynamics and reward func before training
        self._policy_module.set_dynamics_func(self._dynamics_func)
        self._policy_module.set_reward_func(self._reward_func)
        self._policy_module._planning_horizon = self._planning_horizon

        policy_step = self._policy_module.train_step(exp, state.policy)

        self._has_been_trained = True

        return policy_step._replace(
            state=PlannerState(policy=policy_step.state),
            info=PlannerInfo(policy=policy_step.info))

    def calc_loss(self, training_info: TrainingInfo):
        policy_loss = self._policy_module.calc_loss(
            training_info=training_info._replace(
                info=training_info.info.policy))

        return LossInfo(
            loss=policy_loss.loss,
            extra=PlannerLossInfo(policy=policy_loss.extra))

    def after_update(self, training_info):
        self._policy_module._update_target()

    # def _trainable_attributes_to_ignore(self):
    #     return ['_policy_module']

    # def _trainable_attributes_to_ignore(self):
    #     return ['']

    def generate_plan(self, time_step: TimeStep, state, epsilon_greedy):
        assert self._reward_func is not None, ("specify reward function "
                                               "before planning")

        assert self._dynamics_func is not None, ("specify dynamics function "
                                                 "before planning")

        # if not self._has_been_trained:
        #     action = torch.rand([
        #         time_step.observation.shape[0], self._num_actions
        #     ]) * (self._upper_bound -
        #           self._lower_bound) + self._lower_bound * 1.0
        #     return action, state
        # Q-based action sequence population generation
        # tf.random.uniform([batch_size, self._population_size, self._solution_dim]
        # ac_q_pop = self._generate_action_sequence(time_step, state,
        #                                           epsilon_greedy)

        # # Q-sampling-based action generation
        # ac_q_pop = self._generate_action_sequence_random_sampling(
        #     time_step, state, epsilon_greedy)
        # # # save np array
        # # ac_q_pop_np = ac_q_pop.numpy()
        # # np.save('./ac_seq.mat', ac_q_pop_np)
        # # # #ac_q_pop = self._generate_action_sequence_random(time_step, state)
        # # obs_np = time_step.observation.numpy()
        # # np.save('./init_obs.mat', obs_np)

        # self._plan_optimizer.set_cost(self._calc_cost_for_action_sequence)
        # opt_action = self._plan_optimizer.obtain_solution(
        #     time_step, state, ac_q_pop)

        # # option 2
        # # simutineously generate action sequence and evaluate
        # opt_action = self._generate_action_sequence_random_sampling(
        #     time_step, state, epsilon_greedy, mode="mix")
        # action = opt_action[:, 0]

        # # option 3 sac actor: single step action generation
        # action, _ = self._get_action_from_Q(
        #     time_step, state.planner, epsilon_greedy=1
        # )  # always perform sampling from the action distribution

        # option 4 multi-step action generation [Q and A variants]
        # opt_action = self._generate_action_sequence_random_sampling(
        #     time_step, state, epsilon_greedy, mode="mix")
        # action = opt_action[:, 0]

        # # option 5 do action optimization
        # action, planner_state = self._get_action_from_A_optimization(
        #     time_step, state.planner)

        # # option 6:
        # action, planner_state = self._get_action_multi_step_optimization(
        #     time_step, state, self._dynamics_func,  H=1)

        # option 7 NAF
        # opt_action = self._generate_action_sequence_random_sampling(
        #     time_step, state, epsilon_greedy, mode="mix")
        # action = opt_action[:, 0]

        # 7.1
        # action, _ = self._get_action_from_Q(
        #     time_step, state.planner, epsilon_greedy=1
        # )  # always perform sampling from the action distribution

        # option 8 DDPG
        opt_action = self._generate_action_sequence_random_sampling(
            time_step, state, epsilon_greedy, mode="mix")
        action = opt_action[:, 0]

        # option 9
        # action, planner_state = self._get_action_from_Q(time_step,
        #                                                 state.planner,
        #                                                 epsilon_greedy=1)

        # # add epsilon greedy
        # non_greedy_mask = torch.rand(action.shape[0]) < epsilon_greedy

        # # random action
        # action_rand = torch.rand(action.shape) * (
        #     self._upper_bound - self._lower_bound) + self._lower_bound * 1.0

        # action[non_greedy_mask] = action_rand[non_greedy_mask]

        # # option 3
        # action, state = self._get_action_from_Q(time_step, state,
        #                                         epsilon_greedy)

        # option 4: beam search
        # ac_q_pop = self._generate_action_sequence_beam_search(
        #     time_step, state, epsilon_greedy)

        # vis_utils.save_to_np(ac_q_pop, './ac_seqs_beam_init.mat')
        # vis_utils.save_to_np(time_step.observation, './obs_seqs_beam_init.mat')

        action = torch.reshape(action, [time_step.observation.shape[0], -1])
        return action, state

    def rollout(self, time_step: TimeStep, state=None):
        # if self.need_full_rollout_state():
        #     raise NotImplementedError("Storing RNN state to replay buffer "
        #                               "is not supported by DdpgAlgorithm")
        return self.generate_plan(time_step, state)

    def _get_action_from_Q(self, time_step: TimeStep, state, epsilon_greedy):
        """
        Returns:
            state: planner state
        """

        policy_step = self._policy_module.predict_step(time_step, state.policy,
                                                       epsilon_greedy)

        action = policy_step.output

        return action, PlannerState(policy=policy_step.state)

    def _get_action_from_Q_sampling(self,
                                    org_batch_size,
                                    time_step: TimeStep,
                                    state,
                                    mode="SAC"):
        """ Sampling-based approach for select next action
            mode: SAC, NAF, DDPG
        Returns:
            state: planner state
        """
        obs_pop = time_step.observation  # obs has already be expanded

        # batch size after expansion
        batch_size = obs_pop.shape[0]
        pop_size = batch_size // org_batch_size

        solution_size = self._num_actions  # one-step horizon

        # expand
        obs_pop = torch.repeat_interleave(obs_pop, self._repeat_times, dim=0)
        ac_rand_pop = torch.rand(
            batch_size * self._repeat_times, solution_size
        ) * (self._upper_bound - self._lower_bound) + self._lower_bound * 1.0

        critic_input = (obs_pop, ac_rand_pop)

        # # # option 3 sac actor
        # action0, _ = self._get_action_from_Q(
        #     time_step._replace(observation=obs_pop), state, epsilon_greedy=1
        # )  # always perform sampling from the action distribution
        # critic_input0 = (obs_pop, action0)

        # init an empty action for returning, indicating terminate
        action = []

        #mode = "NAF"  #"SAC"
        if self._step_eval_func is not None:
            disagreement = self._step_eval_func(*critic_input)

            critic1, critic_state = self._policy_module._critic_networks.get_preds_mean(
                critic_input)
            critic = critic1 + 0 * disagreement
        elif mode == "SAC":
            # critic0, critic_state0 = self._policy_module._critic_networks.get_preds_mean(
            #     critic_input)
            # critic, critic_state = self._policy_module._critic_networks(
            #     critic_input)
            # critics0, _ = self._policy_module._critic_networks.get_preds(
            #     critic_input0)

            critics, critic_state0 = self._policy_module._critic_networks.get_preds(
                critic_input)
            c_mean = tensor_utils.list_mean(critics)
            c_std = tensor_utils.list_std(critics)
            critic = c_mean
            # if c_std.max() > 5e-4:
            #     return action, state
        elif mode == "DDPG":
            critic, critic_state0 = self._policy_module._get_q_value(
                critic_input)
        elif mode == "NAF":
            critic, critic_state0 = self._policy_module._get_q_value(
                critic_input)

            # if c_std.max() > 5e-4:
            #     return action, state

        # include some diversity mearsure
        # [org_batch_size, expanded_pop]
        critic = critic.reshape(org_batch_size, -1)
        top_k = pop_size
        _, sel_ind = torch.topk(critic, k=top_k)
        #sel_ind = torch.zeros_like(sel_ind)

        # reshape to org batch size*expanded_pop_size*sol_dim
        # and select top-k
        ac_rand_pop = ac_rand_pop.reshape(org_batch_size, -1, solution_size)

        def _batched_index_select(t, dim, inds):
            dummy = inds.unsqueeze(2).expand(
                inds.size(0), inds.size(1), t.size(2))
            out = t.gather(dim, dummy)  # b x e x f
            return out

        action = _batched_index_select(ac_rand_pop, 1, sel_ind).squeeze(1)
        action = action.reshape(-1, solution_size)

        return action, state

    def _get_action_from_Q_sampling_Twin(self, org_batch_size,
                                         time_step: TimeStep, state):
        """ Twin: one for proposal and one for eval
        Returns:
            state: planner state
        """
        obs_pop = time_step.observation  # obs has already be expanded
        mqv1, state1 = self._policy_module._critic_network1(
            (time_step.observation, None), state=state.policy.critic)
        action = mqv1[0]

        # batch size after expansion
        batch_size = obs_pop.shape[0]
        pop_size = batch_size // org_batch_size

        solution_size = self._num_actions  # one-step horizon

        # expand
        obs_pop = torch.repeat_interleave(obs_pop, self._repeat_times, dim=0)
        # obs_std = torch.mean(torch.std(obs_pop, 1))
        # obs_noise = torch.randn_like(obs_pop) * 0.1 * obs_std

        ac_pop = torch.repeat_interleave(action, self._repeat_times, dim=0)
        ac_noise = torch.randn(
            batch_size * self._repeat_times,
            solution_size) * (self._upper_bound - self._lower_bound) * 0.1
        ac_rand_pop = ac_pop + ac_noise

        obs_pop = torch.cat((time_step.observation, obs_pop), 0)
        ac_rand_pop = torch.cat((action, ac_rand_pop), 0)

        critic_input = (obs_pop, ac_rand_pop)

        # # # option 3 sac actor
        # action0, _ = self._get_action_from_Q(
        #     time_step._replace(observation=obs_pop), state, epsilon_greedy=1
        # )  # always perform sampling from the action distribution
        # critic_input0 = (obs_pop, action0)

        # init an empty action for returning, indicating terminate
        action = []

        mqv2, critic_state2 = self._policy_module._critic_network2(
            critic_input, state=state.policy.critic)
        #mqv2, critic_state = self._critic_network2(inputs, state=state)

        critic = mqv2[1].view(-1)

        # if c_std.max() > 5e-4:
        #     return action, state

        # include some diversity mearsure
        # [org_batch_size, expanded_pop]
        critic = critic.reshape(org_batch_size, -1)
        top_k = pop_size
        _, sel_ind = torch.topk(critic, k=top_k)
        #sel_ind = torch.zeros_like(sel_ind)

        # reshape to org batch size*expanded_pop_size*sol_dim
        # and select top-k
        ac_rand_pop = ac_rand_pop.reshape(org_batch_size, -1, solution_size)

        def _batched_index_select(t, dim, inds):
            dummy = inds.unsqueeze(2).expand(
                inds.size(0), inds.size(1), t.size(2))
            out = t.gather(dim, dummy)  # b x e x f
            return out

        action = _batched_index_select(ac_rand_pop, 1, sel_ind).squeeze(1)
        action = action.reshape(-1, solution_size)

        return action, state

    def _get_action_from_A_sampling_with_Q_beam_search(self,
                                                       org_batch_size,
                                                       time_step: TimeStep,
                                                       state,
                                                       epsilon_greedy,
                                                       planning_ind,
                                                       mode="DDPG"):
        """ Action Sampling-based approach for select next action
        Args:
            planning_ind: current planning step index
        Returns:
            state: planner state
        """
        obs_pop_org = time_step.observation  # obs has already be expanded

        # 1) greedy version
        ac_greedy, _ = self._get_action_from_Q(
            time_step._replace(observation=obs_pop_org),
            state,
            epsilon_greedy=0.0
        )  # always perform sampling from the action distribution
        if planning_ind == 0:
            # action_noise = torch.randn_like(ac_greedy) * (
            #     self._upper_bound - self._lower_bound) / 2.0 * 0.1
            # ac_greedy[1:] = ac_greedy[1:] + action_noise[1:]
            ac_rand = torch.rand_like(ac_greedy) * (
                self._upper_bound -
                self._lower_bound) + self._lower_bound * 1.0
            ac_greedy[1:] = ac_rand[1:]

        # batch size after expansion
        batch_size = obs_pop_org.shape[0]
        pop_size = batch_size // org_batch_size

        solution_size = self._num_actions  # one-step horizon

        # expand
        obs_pop = torch.repeat_interleave(
            obs_pop_org, self._repeat_times, dim=0)

        # obs_std = torch.mean(torch.std(obs_pop, 1))
        # obs_noise = torch.randn_like(obs_pop) * 0.1 * obs_std
        # obs_pop = obs_pop + obs_noise

        # obs_pop = torch.cat((obs_pop, time_step.observation), 0)

        # # option 3 sac actor
        ac_rand_pop, _ = self._get_action_from_Q(
            time_step._replace(observation=obs_pop),
            state,
            epsilon_greedy=epsilon_greedy
        )  # always perform sampling from the action distribution

        #critic_input = (obs_pop, ac_rand_pop)

        ac_rand_pop = torch.cat((ac_greedy, ac_rand_pop), 0)
        critic_input = (torch.cat((obs_pop_org, obs_pop), 0), ac_rand_pop)

        # init an empty action for returning, indicating terminate
        action = []

        if mode == "DDPG":
            critic, critic_state = self._policy_module._get_q_value(
                critic_input)
        elif self._step_eval_func is not None:
            disagreement = self._step_eval_func(*critic_input)

            critic1, critic_state = self._policy_module._critic_networks.get_preds_mean(
                critic_input)
            critic = critic1 + 0 * disagreement
        else:
            # critic0, critic_state0 = self._policy_module._critic_networks.get_preds_mean(
            #     critic_input)
            # critic, critic_state = self._policy_module._critic_networks(
            #     critic_input)
            # critics0, _ = self._policy_module._critic_networks.get_preds(
            #     critic_input0)

            critics, critic_state0 = self._policy_module._critic_networks.get_preds(
                critic_input)
            c_mean = tensor_utils.list_mean(critics)
            c_std = tensor_utils.list_std(critics)
            critic = c_mean

        # include some diversity mearsure
        # [org_batch_size, expanded_pop]
        critic = critic.reshape(org_batch_size, -1)
        top_k = pop_size
        _, sel_ind = torch.topk(critic, k=top_k)

        # #diversity-based selection
        # # construct K
        # def _construct_K(X, Y, EK, sigma):
        #     # Args:
        #     # X: [B, d]
        #     n = X.size(0)
        #     m = Y.size(0)
        #     d = X.size(1)

        #     X = X.unsqueeze(1).expand(n, m, d)
        #     Y = Y.unsqueeze(0).expand(n, m, d)

        #     dist = torch.pow(X - Y, 2).sum(2)
        #     K = torch.exp(-0.5 * dist / sigma**2)
        #     K = (EK / torch.trace(K)) * K
        #     return K

        # _X = torch.cat(critic_input, 1)

        # K = _construct_K(_X, _X, 1, 0.1)
        # print(K)
        # #q = sampling_utils.sequential_thinning_dpp_init(K)
        # q = torch.ones(_X.shape[0])
        # q = critic.view(-1)
        # q = q / q.sum()
        # A = sampling_utils.sequential_thinning_dpp_simulation(K, sel_ind.view(-1))
        # print(A.shape)
        # sel_ind = A

        #sel_ind = torch.zeros_like(sel_ind)

        # reshape to org batch size*expanded_pop_size*sol_dim
        # and select top-k
        ac_rand_pop = ac_rand_pop.reshape(org_batch_size, -1, solution_size)

        def _batched_index_select(t, dim, inds):
            dummy = inds.unsqueeze(2).expand(
                inds.size(0), inds.size(1), t.size(2))
            out = t.gather(dim, dummy)  # b x e x f
            return out

        action = _batched_index_select(ac_rand_pop, 1, sel_ind).squeeze(1)
        action = action.reshape(-1, solution_size)

        return action, state

    def _get_action_from_A_sampling(self, org_batch_size, time_step: TimeStep,
                                    state):
        """ Action Sampling-based approach for select next action
        Returns:
            state: planner state
        """
        obs_pop = time_step.observation  # obs has already be expanded

        # batch size after expansion
        batch_size = obs_pop.shape[0]
        pop_size = batch_size // org_batch_size

        solution_size = self._num_actions  # one-step horizon

        # # expand
        # obs_pop = torch.repeat_interleave(obs_pop, self._repeat_times, dim=0)

        # # option 3 sac actor
        action, _ = self._get_action_from_Q(
            time_step._replace(observation=obs_pop), state, epsilon_greedy=1
        )  # always perform sampling from the action distribution

        return action, state

    def _get_action_from_A_optimization(self, time_step: TimeStep, state):
        """ Action optimization-based approach for select next action
        Returns:
            state: mbrl state
        """

        action = self._policy_module.action_optimization(
            time_step, state, iter_num=1)
        return action, state

    def _get_action_multi_step_optimization(self, time_step: TimeStep, state):
        """ Action optimization-based approach for select next action
        Returns:
            state: planner state
        """

        action = self._policy_module.action_optimization_multi_step(
            time_step, state.policy, H=1)
        return action, state

    def _generate_action_sequence_random(self, time_step: TimeStep, state):
        # random population
        # TODO: a future refactor for this and random optimizer
        obs = time_step.observation
        batch_size = obs.shape[0]
        ac_rand_pop = torch.rand(
            batch_size, self._population_size, self._solution_dim
        ) * (self._upper_bound - self._lower_bound) + self._lower_bound * 1.0
        return ac_rand_pop

    def _generate_action_sequence(self, time_step: TimeStep, state,
                                  epsilon_greedy):
        """Generate action sequence proposals according to dynamics and Q.
        The generated action sequences will then be evaluated using the cost.
        [There is a potential that these two steps can be merged together.]
        Args:
            state: mbrl state
        Returns:
            ac_seqs (list): of size [b, p, solution=h*a]
        """

        obs = time_step.observation
        batch_size = obs.shape[0]

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        init_action = self._expand_to_population(time_step.prev_action)
        state = nest.map_structure(self._expand_to_population, state)

        obs = init_obs
        time_step = time_step._replace(
            observation=obs, prev_action=init_action)  # for Q

        # [b, p, h, a]
        ac_seqs = torch.zeros([
            batch_size, self._population_size, self._planning_horizon,
            self._num_actions
        ])

        # merge population with batch
        ac_seqs = ac_seqs.permute(2, 0, 1, 3)
        ac_seqs = torch.reshape(
            ac_seqs, (self._planning_horizon, -1, self._num_actions))

        for i in range(self._planning_horizon):
            action, planner_state = self._get_action_from_Q(
                time_step, state, epsilon_greedy)  # always add noise
            # update policy state part
            state = state._replace(planner=planner_state)
            time_step = time_step._replace(prev_action=action)
            time_step, state = self._dynamics_func(time_step, state)
            ac_seqs[i] = action

        ac_seqs = torch.reshape(ac_seqs, [
            self._planning_horizon, batch_size, self._population_size,
            self._num_actions
        ]).permute(1, 2, 0, 3)
        ac_seqs = torch.reshape(ac_seqs,
                                [batch_size, self._population_size, -1])
        return ac_seqs

    def _expand_to_population(self, data):
        """Expand the input tensor to a population of replications
        Args:
            data (Tensor): input data with shape [batch_size, ...]
        Returns:
            data_population (Tensor) with shape
                                    [batch_size * self._population_size, ...].
            For example data tensor [[a, b], [c, d]] and a population_size of 2,
            we have the following data_population tensor as output
                                    [[a, b], [a, b], [c, d], [c, d]]
        """
        data_population = torch.repeat_interleave(
            data, self._population_size, dim=0)
        return data_population

    def _calc_cost_for_action_sequence(self, time_step: TimeStep, state,
                                       ac_seqs):
        """
        Args:
            time_step (TimeStep): input data for next step prediction
            state (MbrlState): input state for next step prediction
            ac_seqs: action_sequence (Tensor) of shape [batch_size,
                    population_size, solution_dim]), where
                    solution_dim = planning_horizon * num_actions
        Returns:
            cost (tf.Tensor) with shape [batch_size, population_size]
        """
        obs = time_step.observation
        batch_size = obs.shape[0]

        ac_seqs = torch.reshape(
            ac_seqs,
            [batch_size, self._population_size, self._planning_horizon, -1])

        ac_seqs = ac_seqs.permute(2, 0, 1, 3)
        ac_seqs = torch.reshape(
            ac_seqs, (self._planning_horizon, -1, self._num_actions))

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        state = nest.map_structure(self._expand_to_population, state)

        obs = init_obs
        cost = 0
        for i in range(ac_seqs.shape[0]):
            action = ac_seqs[i]
            time_step = time_step._replace(prev_action=action)
            time_step, state = self._dynamics_func(time_step, state)
            next_obs = time_step.observation
            # Note: currently using (next_obs, action), might need to
            # consider (obs, action) in order to be more compatible
            # with the conventional definition of reward function
            reward_step = self._reward_func(obs, action, next_obs)
            cost = cost - reward_step
            obs = next_obs

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])
        return cost

    def _generate_action_sequence_random_sampling(self,
                                                  time_step: TimeStep,
                                                  state,
                                                  epsilon_greedy,
                                                  mode="random"):
        """Generate action sequence proposals according to dynamics and Q by
        random sampling. The generated action sequences will then be evaluated
        using the cost.
        [There is a potential that these two steps can be merged together.]
        Args:
            state: mbrl state
        Returns:
            ac_seqs (list): of size [b, p, solution=h*a]
        """

        obs = time_step.observation
        #print(obs[0, 3:8])
        batch_size = obs.shape[0]

        # some initialization
        # [b, p, h, a]
        ac_seqs = torch.rand([
            batch_size, self._population_size, self._planning_horizon,
            self._num_actions
        ]) * (self._upper_bound - self._lower_bound) + self._lower_bound * 1.0

        ac_seqs_org = ac_seqs.clone()

        obs_seqs = torch.zeros([
            batch_size, self._population_size, self._planning_horizon,
            obs.shape[1]
        ])

        # merge population with batch
        ac_seqs = ac_seqs.permute(2, 0, 1, 3)
        ac_seqs = torch.reshape(
            ac_seqs, (self._planning_horizon, -1, self._num_actions))

        # merge population with batch
        obs_seqs = obs_seqs.permute(2, 0, 1, 3)
        obs_seqs = torch.reshape(obs_seqs,
                                 (self._planning_horizon, -1, obs.shape[1]))
        #----------------------

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        init_action = self._expand_to_population(time_step.prev_action)
        # # randomize first action
        # init_action[1:] = ac_seqs[0, 1:]
        state = nest.map_structure(self._expand_to_population, state)

        obs = init_obs
        time_step = time_step._replace(
            observation=obs, prev_action=init_action)  # for Q

        action_noise = torch.randn_like(init_action) * (
            self._upper_bound - self._lower_bound) / 2.0 * 0.1

        cost = 0
        discount = self._discount
        # TODO: this is related to target value
        terminated = False
        with torch.no_grad():
            for i in range(self._planning_horizon):
                obs_seqs[i] = time_step.observation
                if mode == "random":
                    action = ac_seqs[i]
                elif "mix" in mode:
                    #else:
                    #---Q
                    if not terminated:
                        #1
                        # action, planner_state = self._get_action_from_Q_sampling(
                        #     batch_size, time_step,
                        #     state.planner)  # always add noise
                        # # 2
                        # action, planner_state = self._get_action_from_A_sampling(
                        #     batch_size, time_step,
                        #     state.planner)  # always add noise
                        # update policy state part
                        # # 3
                        # action, planner_state = self._get_action_from_A_optimization(
                        #     time_step,
                        #     state.planner)
                        # 4, NAF, DDPG
                        # action, planner_state = self._get_action_from_Q_sampling(
                        #     batch_size, time_step, state.planner,
                        #     mode="DDPG")  # always add noise

                        # # 5: twin
                        # action, planner_state = self._get_action_from_Q_sampling_Twin(
                        #     batch_size, time_step,
                        #     state.planner)  # always add noise
                        # 6 use sampler
                        action, planner_state = self._get_action_from_A_sampling_with_Q_beam_search(
                            batch_size,
                            time_step,
                            state.planner,
                            epsilon_greedy,
                            i,
                            mode="DDPG")
                        # if i == 0 and action.shape[0] > 1:
                        #     # action[1:] = ac_seqs[i, 1:]
                        #     action[1:] = action[1:] + action_noise[1:]

                        # debug
                        # action, planner_state = self._get_action_from_Q(
                        #     time_step,
                        #     state.planner,
                        #     epsilon_greedy)
                        if len(action) == 0:
                            action = ac_seqs[i]
                            terminated = True
                    else:
                        action = ac_seqs[i]

                    ac_seqs[i] = action

                with alf.summary.scope("prop_actions"):
                    safe_mean_hist_summary("prop_actions", action, None)

                time_step = time_step._replace(prev_action=action)
                time_step, state = self._dynamics_func(time_step, state)

                # immediate evaluation using reward function
                next_obs = time_step.observation
                #cur_obs = obs_seqs[i]
                #reward_step = self._reward_func(obs, action, next_obs)
                reward_step = self._reward_func(next_obs, action)
                obs = next_obs
                reward_step = reward_step.reshape(-1, 1)
                cost = cost - discount * reward_step
                discount *= discount
        # further add terminal values to the cost with the learned value func
        with torch.no_grad():
            # q_action, planner_state = self._get_action_from_Q(
            #     time_step, state, epsilon_greedy=0)  # always add noise
            # q_action, planner_state = self._get_action_from_Q_sampling(
            #     batch_size, time_step, state.planner)  # always add noise
            # critic_input = (time_step.observation, q_action)
            # critic_compare, critic_state = self._policy_module._critic_networks.get_preds_max(
            #     critic_input)

            # critic = self._policy_module.cal_value(
            #     time_step, state.planner.policy, flag="mean")

            # critic_std = self._policy_module.cal_value(
            #     time_step, state.planner.policy, flag="std")
            # critic = critic_mean + critic_std * 10.0
            # critic = self._policy_module.cal_value(
            #     time_step, state.planner.policy, flag="softmax")
            # NAF
            # critic, _ = self._policy_module._get_state_value(
            #     (time_step.observation, None), state.planner.policy)
            # DDPG
            critic, _ = self._policy_module._get_state_value(
                time_step.observation, state.planner.policy)

            # this is required for all
            critic = critic.reshape(-1, 1)

            cost = cost - discount * critic
            with alf.summary.scope("terminal_critic"):
                safe_mean_hist_summary("terminal_critic", critic.view(-1),
                                       None)

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])
        # the action sequences are unnecessary now
        ac_seqs = torch.reshape(ac_seqs, [
            self._planning_horizon, batch_size, self._population_size,
            self._num_actions
        ]).permute(1, 2, 0, 3)
        # ac_seqs = torch.reshape(ac_seqs,
        #                         [batch_size, self._population_size, -1])

        # [B, P, H, D]
        obs_seqs = torch.reshape(
            obs_seqs,
            [self._planning_horizon, batch_size, self._population_size, -1
             ]).permute(1, 2, 0, 3)
        # obs_seqs = torch.reshape(obs_seqs,
        #                          [batch_size, self._population_size, -1])

        # vis_utils.save_to_np(ac_seqs, './ac_seqs_std_car_before_train.mat')
        # vis_utils.save_to_np(obs_seqs, './obs_seqs_std_car_before_train.mat')

        min_ind = torch.argmin(cost, dim=-1).long()
        # TODO: need to check if batch_index_select is needed
        opt_ac_seqs = ac_seqs.index_select(1, min_ind).squeeze(1)
        #opt_ac_seqs = ac_seqs_org.index_select(1, min_ind).squeeze(1)

        return opt_ac_seqs

    def _generate_action_sequence_beam_search(self, time_step: TimeStep, state,
                                              epsilon_greedy):
        """Generate action sequence proposals according to dynamics and Q by
        random sampling. The generated action sequences will then be evaluated
        using the cost.
        [There is a potential that these two steps can be merged together.]
        Args:
            state: mbrl state
        Returns:
            ac_seqs (list): of size [b, p, solution=h*a]
        """

        obs = time_step.observation
        batch_size = obs.shape[0]
        state = state._replace(dynamics=state.dynamics._replace(feature=obs))

        # batch size, pop_size, horizon, action_dim
        ac_seqs = beam_search.beam_decode(
            time_step,
            state,
            self._policy_module._critic_network1,
            self._dynamics_func,
            max_len=self._planning_horizon,
            number_required=self._population_size,
            lower_bound=int(self._lower_bound),
            upper_bound=int(self._upper_bound),
            solution_size=self._num_actions)

        # the action sequences are unnecessary now
        # ac_seqs = torch.reshape(ac_seqs, [
        #     self._planning_horizon, batch_size, self._population_size,
        #     self._num_actions
        # ]).permute(1, 2, 0, 3)
        ac_seqs = torch.reshape(ac_seqs, [
            batch_size, self._population_size, self._planning_horizon,
            self._num_actions
        ])

        return ac_seqs

    def _generate_action_sequence_random_sampling_simutaneous(
            self, time_step: TimeStep, state, epsilon_greedy):
        """Generate action sequence proposals according to dynamics and Q by
        random sampling. The generated action sequences will then be evaluated
        using the cost.
        [There is a potential that these two steps can be merged together.]
        Args:
            state: mbrl state
        Returns:
            ac_seqs (list): of size [b, p, solution=h*a]
        """

        obs = time_step.observation
        batch_size = obs.shape[0]

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        init_action = self._expand_to_population(time_step.prev_action)
        state = nest.map_structure(self._expand_to_population, state)

        obs = init_obs
        time_step = time_step._replace(
            observation=obs, prev_action=init_action)  # for Q

        # [b, p, h, a]
        ac_seqs = torch.zeros([
            batch_size, self._population_size, self._planning_horizon,
            self._num_actions
        ])

        # obs
        obs_seqs = torch.zeros([
            batch_size, self._population_size, self._planning_horizon,
            obs.shape[1]
        ])

        # merge population with batch
        ac_seqs = ac_seqs.permute(2, 0, 1, 3)
        ac_seqs = torch.reshape(
            ac_seqs, (self._planning_horizon, -1, self._num_actions))

        # merge population with batch
        obs_seqs = obs_seqs.permute(2, 0, 1, 3)
        obs_seqs = torch.reshape(obs_seqs,
                                 (self._planning_horizon, -1, obs.shape[1]))

        cost = 0
        for i in range(self._planning_horizon):
            obs_seqs[i] = time_step.observation
            action, planner_state = self._get_action_from_Q_sampling(
                time_step, state)  # always add noise
            # update policy state part
            state = state._replace(planner=planner_state)
            time_step = time_step._replace(prev_action=action.detach())
            time_step, state = self._dynamics_func(time_step, state)
            ac_seqs[i] = action

            # immediate evaluation using reward function
            next_obs = time_step.observation
            reward_step = self._reward_func(next_obs, action)
            cost = cost - reward_step

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])

        # the action sequences are unnecessary now
        ac_seqs = torch.reshape(ac_seqs, [
            self._planning_horizon, batch_size, self._population_size,
            self._num_actions
        ]).permute(1, 2, 0, 3)
        # ac_seqs = torch.reshape(ac_seqs,
        #                         [batch_size, self._population_size, -1])

        # [B, P, H, D]
        obs_seqs = torch.reshape(
            obs_seqs,
            [self._planning_horizon, batch_size, self._population_size, -1
             ]).permute(1, 2, 0, 3)
        # obs_seqs = torch.reshape(obs_seqs,
        #                          [batch_size, self._population_size, -1])

        # vis_utils.save_to_np(ac_seqs, './ac_seqs_random.mat')
        # vis_utils.save_to_np(obs_seqs, './obs_seqs_random.mat')

        min_ind = torch.argmin(cost, dim=-1).long()
        # TODO: need to check if batch_index_select is needed
        opt_ac_seqs = ac_seqs.index_select(1, min_ind).squeeze(1)

        return opt_ac_seqs
