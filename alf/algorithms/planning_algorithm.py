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

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 TimeStep, TrainingInfo)
from alf.nest import nest
from alf.optimizers.random import RandomOptimizer, QOptimizer

from alf.utils import vis_utils

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
        return cost

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
        policy_step = self._policy_module.train_step(exp, state.policy)

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

    def _trainable_attributes_to_ignore(self):
        return ['']

    def generate_plan(self, time_step: TimeStep, state, epsilon_greedy):
        assert self._reward_func is not None, ("specify reward function "
                                               "before planning")

        assert self._dynamics_func is not None, ("specify dynamics function "
                                                 "before planning")

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
        # action = opt_action[:, 0]

        # simutineously generate action sequence and evaluate
        opt_ac_seqs = self._generate_action_sequence_random_sampling(
            time_step, state, epsilon_greedy)
        action = opt_ac_seqs[:, 0]

        # # option 3
        # action, state = self._get_action_from_Q(time_step, state,
        #                                         epsilon_greedy)

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

        policy_step = self._policy_module.predict_step(
            time_step, state.planner.policy, epsilon_greedy)

        action = policy_step.output

        return action, PlannerState(policy=policy_step.state)

    def _get_action_from_Q_sampling(self, time_step: TimeStep, state):
        """ Sampling-based approach for select next action
        Returns:
            state: planner state
        """
        obs_pop = time_step.observation  # obs has already be expanded

        batch_size = obs_pop.shape[0]

        solution_size = self._num_actions  # one-step horizon

        # expand
        repeat_times = 10
        obs_pop = torch.repeat_interleave(obs_pop, repeat_times, dim=0)
        ac_rand_pop = torch.rand(batch_size * repeat_times, solution_size) * (
            self._upper_bound - self._lower_bound) + self._lower_bound * 1.0

        critic_input = (obs_pop, ac_rand_pop)

        critic, critic_state = self._policy_module._critic_network1(
            critic_input)

        critic = critic.reshape(batch_size, repeat_times)
        top_k = 1
        _, sel_ind = torch.topk(critic, k=min(top_k, critic.shape[0]))

        ac_rand_pop = ac_rand_pop.reshape(batch_size, repeat_times, -1)

        def _batched_index_select(t, dim, inds):
            dummy = inds.unsqueeze(2).expand(
                inds.size(0), inds.size(1), t.size(2))
            out = t.gather(dim, dummy)  # b x e x f
            return out

        action = _batched_index_select(ac_rand_pop, 1, sel_ind).squeeze(1)

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
            reward_step = self._reward_func(next_obs, action)
            cost = cost - reward_step
            obs = next_obs

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])
        return cost

    def _generate_action_sequence_random_sampling(self, time_step: TimeStep,
                                                  state, epsilon_greedy):
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
            time_step = time_step._replace(prev_action=action)
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

        # vis_utils.save_to_np(ac_seqs, './ac_seqs_latest.mat')
        # vis_utils.save_to_np(obs_seqs, './obs_seqs_latest.mat')

        min_ind = torch.argmin(cost, dim=-1).long()
        # TODO: need to check if batch_index_select is needed
        opt_ac_seqs = ac_seqs.index_select(1, min_ind).squeeze(1)

        return opt_ac_seqs
