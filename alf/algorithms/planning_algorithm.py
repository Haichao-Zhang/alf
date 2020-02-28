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

import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.utils import common as tfa_common

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec
from tf_agents.trajectories.policy_step import PolicyStep

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.data_structures import ActionTimeStep, namedtuple, Experience, TrainingInfo
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.optimizers.random import RandomOptimizer, QOptimizer
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.utils import losses, common

DdpgCriticState = namedtuple("DdpgCriticState",
                             ['critic', 'target_actor', 'target_critic'])
DdpgCriticInfo = namedtuple("DdpgCriticInfo", ["q_value", "target_q_value"])
DdpgActorState = namedtuple("DdpgActorState", ['actor', 'critic'])

PlannerState = namedtuple(
    "PlannerState", ["actor", "critic"], default_value=())
PlannerInfo = namedtuple("PlannerInfo",
                         ["action_distribution", "actor_loss", "critic"])

PlannerLossInfo = namedtuple('PlannerLossInfo', ('actor', 'critic'))


@gin.configurable
class PlanAlgorithm(OffPolicyAlgorithm):
    """Planning Module

    This module plans for actions based on initial observation
    and specified reward and dynamics functions
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 planning_horizon,
                 train_state_spec=None,
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

        flat_action_spec = tf.nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        assert not tensor_spec.is_discrete(action_spec), "only support \
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

    def train_step(self, time_step: ActionTimeStep, state):
        """
        Args:
            time_step (ActionTimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
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
            next_time_step (ActionTimeStep) and the next_state
        """
        self._dynamics_func = dynamics_func

    def generate_plan(self, time_step: ActionTimeStep, state):
        """Compute the plan based on the provided observation and action
        Args:
            time_step (ActionTimeStep): input data for next step prediction
            state: input state next step prediction
        Returns:
            action: planned action for the given inputs
        """
        pass

    def calc_loss(self, info):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(
            loss=info.loss, scalar_loss=loss.loss, extra=loss.extra)


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
            planning_horizon=planning_horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, ("RandomShootingAlgorithm doesn't "
                                            "support nested action_spec")

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(flat_feature_spec) == 1, ("RandomShootingAlgorithm doesn't "
                                             "support nested feature_spec")

        self._population_size = population_size
        solution_size = self._planning_horizon * self._num_actions
        self._plan_optimizer = RandomOptimizer(
            solution_size,
            self._population_size,
            upper_bound=action_spec.maximum,
            lower_bound=action_spec.minimum)

    def train_step(self, time_step: ActionTimeStep, state):
        """
        Args:
            time_step (ActionTimeStep): input data for planning
            state: state for planning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        return AlgorithmStep(outputs=(), state=(), info=())

    def generate_plan(self, time_step: ActionTimeStep, state):
        assert self._reward_func is not None, ("specify reward function "
                                               "before planning")

        assert self._dynamics_func is not None, ("specify dynamics function "
                                                 "before planning")

        self._plan_optimizer.set_cost(self._calc_cost_for_action_sequence)
        opt_action = self._plan_optimizer.obtain_solution(time_step, state)
        action = opt_action[:, 0]
        action = tf.reshape(action, [time_step.observation.shape[0], -1])
        return action, state

    def _expand_to_population(self, data):
        """Expand the input tensor to a population of replications
        Args:
            data (tf.Tensor): input data with shape [batch_size, ...]
        Returns:
            data_population (tf.Tensor) with shape
                                    [batch_size * self._population_size, ...].
            For example data tensor [[a, b], [c, d]] and a population_size of 2,
            we have the following data_population tensor as output
                                    [[a, b], [a, b], [c, d], [c, d]]
        """
        data_population = tf.tile(
            tf.expand_dims(data, 1),
            [1, self._population_size] + [1] * len(data.shape[1:]))
        data_population = tf.reshape(data_population,
                                     [-1] + data.shape[1:].as_list())
        return data_population

    def _calc_cost_for_action_sequence(self, time_step: ActionTimeStep, state,
                                       ac_seqs):
        """
        Args:
            time_step (ActionTimeStep): input data for next step prediction
            state (MbrlState): input state for next step prediction
            ac_seqs: action_sequence (tf.Tensor) of shape [batch_size,
                    population_size, solution_dim]), where
                    solution_dim = planning_horizon * num_actions
        Returns:
            cost (tf.Tensor) with shape [batch_size, population_size]
        """
        obs = time_step.observation
        batch_size = obs.shape[0]
        init_costs = tf.zeros([batch_size, self._population_size])
        ac_seqs = tf.reshape(
            ac_seqs,
            [batch_size, self._population_size, self._planning_horizon, -1])
        ac_seqs = tf.reshape(
            tf.transpose(ac_seqs, [2, 0, 1, 3]),
            [self._planning_horizon, -1, self._num_actions])

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        state = tf.nest.map_structure(self._expand_to_population, state)

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
        cost = tf.reshape(cost, [batch_size, -1])
        return cost


@gin.configurable
class QShootingAlgorithm(PlanAlgorithm):
    """Q-value guided planning method.
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 population_size,
                 planning_horizon,
                 actor_network: Network,
                 critic_network: Network,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 critic_loss=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 upper_bound=None,
                 lower_bound=None,
                 hidden_size=256,
                 debug_summaries=False,
                 name="QShootingAlgorithm"):
        """Create a QShootingAlgorithm.

        Args:
            population_size (int): the size of polulation for Q shooting
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
            hidden_size (int|tuple): size of hidden layer(s)
        """
        train_state_spec = PlannerState(
            actor=DdpgActorState(
                actor=actor_network.state_spec,
                critic=critic_network.state_spec),
            critic=DdpgCriticState(
                critic=critic_network.state_spec,
                target_actor=actor_network.state_spec,
                target_critic=critic_network.state_spec))
        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            planning_horizon=planning_horizon,
            train_state_spec=train_state_spec,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, ("QShootingAlgorithm doesn't "
                                            "support nested action_spec")

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(flat_feature_spec) == 1, ("QShootingAlgorithm doesn't "
                                             "support nested feature_spec")

        self._population_size = population_size
        solution_size = self._planning_horizon * self._num_actions
        self._plan_optimizer = QOptimizer(
            solution_size,
            self._population_size,
            upper_bound=action_spec.maximum,
            lower_bound=action_spec.minimum)

        # for Q-network learning
        self._actor_network = actor_network
        self._critic_network = critic_network
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer

        self._target_actor_network = actor_network.copy(
            name='target_actor_network')
        self._target_critic_network = critic_network.copy(
            name='target_critic_network')

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping

        if critic_loss is None:
            critic_loss = OneStepTDLoss(debug_summaries=debug_summaries)
        self._critic_loss = critic_loss

        self._ou_process = create_ou_process(action_spec, ou_stddev,
                                             ou_damping)

        self._update_target = common.get_target_updater(
            models=[self._actor_network, self._critic_network],
            target_models=[
                self._target_actor_network, self._target_critic_network
            ],
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

    def _critic_train_step(self, exp: Experience, state: DdpgCriticState):
        target_action, target_actor_state = self._target_actor_network(
            exp.observation,
            step_type=exp.step_type,
            network_state=state.target_actor)
        target_q_value, target_critic_state = self._target_critic_network(
            (exp.observation, target_action),
            step_type=exp.step_type,
            network_state=state.target_critic)

        q_value, critic_state = self._critic_network(
            (exp.observation, exp.action),
            step_type=exp.step_type,
            network_state=state.critic)

        state = DdpgCriticState(
            critic=critic_state,
            target_actor=target_actor_state,
            target_critic=target_critic_state)

        info = DdpgCriticInfo(q_value=q_value, target_q_value=target_q_value)

        return state, info

    def _actor_train_step(self, exp: Experience, state: DdpgActorState):
        action, actor_state = self._actor_network(
            exp.observation, exp.step_type, network_state=state.actor)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(action)
            q_value, critic_state = self._critic_network(
                (exp.observation, action), network_state=state.critic)

        dqda = tape.gradient(q_value, action)

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = tf.clip_by_value(dqda, -self._dqda_clipping,
                                        self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                tf.stop_gradient(dqda + action), action)
            loss = tf.reduce_sum(loss, axis=list(range(1, len(loss.shape))))
            return loss

        actor_loss = tf.nest.map_structure(actor_loss_fn, dqda, action)
        state = DdpgActorState(actor=actor_state, critic=critic_state)
        info = LossInfo(
            loss=tf.add_n(tf.nest.flatten(actor_loss)), extra=actor_loss)
        return PolicyStep(action=action, state=state, info=info)

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
        critic_state, critic_info = self._critic_train_step(
            exp=exp, state=state.critic)
        policy_step = self._actor_train_step(exp=exp, state=state.actor)
        return policy_step._replace(
            state=PlannerState(actor=policy_step.state, critic=critic_state),
            info=PlannerInfo(
                action_distribution=policy_step.action,
                critic=critic_info,
                actor_loss=policy_step.info))

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss = self._critic_loss(
            training_info=training_info,
            value=training_info.info.critic.q_value,
            target_value=training_info.info.critic.target_q_value)

        actor_loss = training_info.info.actor_loss

        return LossInfo(
            loss=critic_loss.loss + actor_loss.loss,
            extra=PlannerLossInfo(
                critic=critic_loss.extra, actor=actor_loss.extra))

    def after_train(self, training_info):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return ['_target_actor_network', '_target_critic_network']

    def generate_plan(self, time_step: ActionTimeStep, state):
        assert self._reward_func is not None, ("specify reward function "
                                               "before planning")

        assert self._dynamics_func is not None, ("specify dynamics function "
                                                 "before planning")

        # Q-based action sequence population generation
        # tf.random.uniform([batch_size, self._population_size, self._solution_dim]
        ac_q_pop = self._generate_action_sequence(time_step, state)
        self._plan_optimizer.set_cost(self._calc_cost_for_action_sequence)
        opt_action = self._plan_optimizer.obtain_solution(
            time_step, state, ac_q_pop)
        action = opt_action[:, 0]
        action = tf.reshape(action, [time_step.observation.shape[0], -1])
        return action, state

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                mode=RLAlgorithm.ROLLOUT):
        # if self.need_full_rollout_state():
        #     raise NotImplementedError("Storing RNN state to replay buffer "
        #                               "is not supported by DdpgAlgorithm")
        return self.generate_plan(time_step, state)

    def _get_action_from_Q(self, time_step: ActionTimeStep, state,
                           epsilon_greedy):
        action, state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=())  # temp solution
        empty_state = tf.nest.map_structure(lambda x: (),
                                            self.train_state_spec)

        def _sample(a, ou):
            return tf.cond(
                tf.less(tf.random.uniform((), 0, 1),
                        epsilon_greedy), lambda: a + ou(), lambda: a)

        noisy_action = tf.nest.map_structure(_sample, action, self._ou_process)
        noisy_action = tf.nest.map_structure(tfa_common.clip_to_spec,
                                             noisy_action, self._action_spec)

        return noisy_action

    def _generate_action_sequence(self, time_step: ActionTimeStep, state):
        """Generate action sequence proposals according to dynamics and Q.
        The generated action sequences will then be evaluated using the cost.
        [There is a potential that these two steps can be merged together.]

        Returns:
            ac_seqs (list): of size [b, p, solution=h*a]
        """

        obs = time_step.observation
        batch_size = obs.shape[0]

        # [b, p, h, a]
        ac_seqs = tf.zeros([
            batch_size, self._population_size, self._planning_horizon,
            self._num_actions
        ])

        # merge population with batch
        ac_seqs = tf.reshape(
            tf.transpose(ac_seqs, [2, 0, 1, 3]),
            [self._planning_horizon, -1, self._num_actions])

        ac_seqs_np = ac_seqs.numpy()

        # # get the action for the first time step to start with
        # ac_seqs[i] = self._get_action_from_Q(time_step, state, 0.1)

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        state = tf.nest.map_structure(self._expand_to_population, state)

        obs = init_obs
        time_step = time_step._replace(observation=obs)  # for Q

        # convert to tf loop
        for i in range(ac_seqs.shape[0]):  # time step
            # time_step: obs for Q; prev_feature for dynamics
            action = self._get_action_from_Q(time_step, state, 0.1)
            ac_seqs_np[i] = action.numpy()
            time_step = time_step._replace(prev_action=action)
            time_step, state = self._dynamics_func(time_step, state)
        ac_seqs = tf.convert_to_tensor(ac_seqs_np, dtype=tf.float32)

        # def _train_loop_body():

        # num_steps = ac_seqs.shape[0]
        # [_, _, info_ta] = tf.while_loop(
        #         cond=lambda counter, *_: tf.less(counter, num_steps),
        #         body=_train_loop_body,
        #         loop_vars=[counter, first_train_state, info_ta],
        #         back_prop=True,
        #         name="train_loop")

        ac_seqs = tf.transpose(
            tf.reshape(ac_seqs, [
                self._planning_horizon, batch_size, self._population_size,
                self._num_actions
            ]), [1, 2, 0, 3])
        ac_seqs = tf.reshape(ac_seqs, [batch_size, self._population_size, -1])
        return ac_seqs

    def _expand_to_population(self, data):
        """Expand the input tensor to a population of replications
        Args:
            data (tf.Tensor): input data with shape [batch_size, ...]
        Returns:
            data_population (tf.Tensor) with shape
                                    [batch_size * self._population_size, ...].
            For example data tensor [[a, b], [c, d]] and a population_size of 2,
            we have the following data_population tensor as output
                                    [[a, b], [a, b], [c, d], [c, d]]
        """
        data_population = tf.tile(
            tf.expand_dims(data, 1),
            [1, self._population_size] + [1] * len(data.shape[1:]))
        data_population = tf.reshape(data_population,
                                     [-1] + data.shape[1:].as_list())
        return data_population

    def _calc_cost_for_action_sequence(self, time_step: ActionTimeStep, state,
                                       ac_seqs):
        """
        Args:
            time_step (ActionTimeStep): input data for next step prediction
            state (MbrlState): input state for next step prediction
            ac_seqs: action_sequence (tf.Tensor) of shape [batch_size,
                    population_size, solution_dim]), where
                    solution_dim = planning_horizon * num_actions
        Returns:
            cost (tf.Tensor) with shape [batch_size, population_size]
        """
        obs = time_step.observation
        batch_size = obs.shape[0]
        init_costs = tf.zeros([batch_size, self._population_size])
        ac_seqs = tf.reshape(
            ac_seqs,
            [batch_size, self._population_size, self._planning_horizon, -1])
        ac_seqs = tf.reshape(
            tf.transpose(ac_seqs, [2, 0, 1, 3]),
            [self._planning_horizon, -1, self._num_actions])

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        state = tf.nest.map_structure(self._expand_to_population, state)

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
        cost = tf.reshape(cost, [batch_size, -1])
        return cost


def create_ou_process(action_spec, ou_stddev, ou_damping):
    """Create nested zero-mean Ornstein-Uhlenbeck processes.

    The temporal update equation is:
    `x_next = (1 - damping) * x + N(0, std_dev)`

    Args:
        action_spec (nested BountedTensorSpec): action spec
        ou_damping (float): Damping rate in the above equation. We must have
            0 <= damping <= 1.
        ou_stddev (float): Standard deviation of the Gaussian component.
    Returns:
        nested OUProcess with the same structure as action_spec.
    """
    # todo with seed None
    seed_stream = tfp.util.SeedStream(seed=None, salt='ou_noise')

    def _create_ou_process(action_spec):
        return tfa_common.OUProcess(
            lambda: tf.zeros(action_spec.shape, dtype=action_spec.dtype),
            ou_damping,
            ou_stddev,
            seed=seed_stream())

    ou_process = tf.nest.map_structure(_create_ou_process, action_spec)
    return ou_process
