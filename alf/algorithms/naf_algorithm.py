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
"""Normalized Advantage Function (NAF)."""

import numpy as np
import gin
import functools

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, TrainingInfo
from alf.nest import nest
from alf.networks import NafCriticNetwork, ActorNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, spec_utils

NafCriticState = namedtuple(
    "NafCriticState",
    ['critic1', 'critic2', 'target_actor', 'target_critic1', 'target_critic2'])
NafCriticInfo = namedtuple(
    "NafCriticInfo",
    ["q_value1", "q_value2", "target_q_value1", "target_q_value2"])
NafActorState = namedtuple("NafActorState", ['actor', 'critic1', 'critic2'])
NafState = namedtuple("NafState", ['actor', 'critic'])
NafInfo = namedtuple(
    "NafInfo", ["action_distribution", "actor_loss", "critic"],
    default_value=())
NafLossInfo = namedtuple('NafLossInfo', ('actor', 'critic1', 'critic2'))


@gin.configurable
class NafAlgorithm(OffPolicyAlgorithm):
    """Normalized Advantage Function (NAF).

    Reference:
    Gu et al "Continuous Deep Q-Learning with Model-based Acceleration"
    https://arxiv.org/pdf/1603.00748.pdf
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 actor_network: ActorNetwork,
                 critic_network: NafCriticNetwork,
                 env=None,
                 config: TrainerConfig = None,
                 ou_scale=1.0,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 critic_loss_ctor=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 gradient_clipping=None,
                 debug_summaries=False,
                 name="NafAlgorithm"):
        """Create a NafAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            critic_network (Network): The network will be called with
                call(observation, action).
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            ou_stddev (float): Standard deviation for the Ornstein-Uhlenbeck
                (OU) noise added in the default collect policy.
            ou_damping (float): Damping factor for the OU noise added in the
                default collect policy.
            critic_loss_ctor (None|OneStepTDLoss): an object for calculating critic
                loss. If None, a default OneStepTDLoss will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            gradient_clipping (float): Norm length to clip gradients.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        critic_network1 = critic_network.copy()
        critic_network2 = critic_network.copy()

        train_state_spec = NafState(
            actor=NafActorState(
                actor=actor_network.state_spec,
                critic1=critic_network.state_spec,
                critic2=critic_network.state_spec),
            critic=NafCriticState(
                critic1=critic_network.state_spec,
                critic2=critic_network.state_spec,
                target_actor=actor_network.state_spec,
                target_critic1=critic_network.state_spec,
                target_critic2=critic_network.state_spec))

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=train_state_spec,
            env=env,
            config=config,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)

        if critic_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])

        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer,
                               [critic_network1, critic_network2])

        self._critic_network1 = critic_network1
        self._critic_network2 = critic_network2
        self._actor_network = actor_network

        self._target_critic_network1 = critic_network.copy()
        self._target_critic_network2 = critic_network.copy()
        self._target_actor_network = actor_network.copy()

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping

        if critic_loss_ctor is None:
            critic_loss_ctor = functools.partial(
                OneStepTDLoss, debug_summaries=debug_summaries)
        self._critic_loss1 = critic_loss_ctor(name="critic_loss1")
        self._critic_loss2 = critic_loss_ctor(name="critic_loss2")

        self._ou_process = common.create_ou_process(action_spec, ou_scale,
                                                    ou_stddev, ou_damping)

        self._update_target = common.get_target_updater(
            models=[
                self._actor_network, self._critic_network1,
                self._critic_network2
            ],
            target_models=[
                self._target_actor_network, self._target_critic_network1,
                self._target_critic_network2
            ],
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=1.):
        action, state = self._actor_network(
            time_step.observation, state=state.actor.actor)

        empty_state = nest.map_structure(lambda x: (), self.train_state_spec)

        def _sample(a, ou):
            if torch.rand(1) < epsilon_greedy:
                return a + ou()
            else:
                return a

        noisy_action = nest.map_structure(_sample, action, self._ou_process)
        noisy_action = nest.map_structure(spec_utils.clip_to_spec,
                                          noisy_action, self._action_spec)

        state = empty_state._replace(
            actor=NafActorState(actor=state, critic1=(), critic2=()))
        return AlgStep(
            output=noisy_action,
            state=state,
            info=NafInfo(action_distribution=action))

    def rollout_step(self, time_step: TimeStep, state=None):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by NafAlgorithm")
        return self.predict_step(time_step, state, epsilon_greedy=1.0)

    def _critic_train_step(self, exp: Experience, state: NafCriticState):
        target_action, target_actor_state = self._target_actor_network(
            exp.observation, state=state.target_actor)

        # swap action
        target_q_value1, target_critic1_state = self._target_critic_network1(
            (exp.observation, target_action), state=state.target_critic1)
        target_q_value2, target_critic2_state = self._target_critic_network2(
            (exp.observation, target_action), state=state.target_critic2)

        # target_q_value1 = mqv_target1[2].view(-1)
        # target_q_value2 = mqv_target2[2].view(-1)

        # # BUG, should use [1] which is the Q value
        # target_q_value1 = mqv_target1[1].view(-1)
        # target_q_value2 = mqv_target2[1].view(-1)

        #target_q_value = torch.min(target_q_value1, target_q_value2)

        q_value1, critic1_state = self._critic_network1(
            (exp.observation, exp.action), state=state.critic1)

        q_value2, critic2_state = self._critic_network2(
            (exp.observation, exp.action), state=state.critic2)

        # q_value1 = mqv1[1].view(-1)
        # q_value2 = mqv2[1].view(-1)

        # action_target1 = mqv1[0]  # SAC style

        # def _sample(a, scale=1.0):
        #     return a + torch.randn_like(a) * (
        #         self._action_spec.maximum -
        #         self._action_spec.minimum) / 2. * scale

        # noisy_action1 = nest.map_structure(_sample, action_target1, 0.1)
        # noisy_action1 = nest.map_structure(spec_utils.clip_to_spec,
        #                                    noisy_action1, self._action_spec)

        # noisy_action2 = nest.map_structure(_sample, action_target2, 0.01)
        # noisy_action2 = nest.map_structure(spec_utils.clip_to_spec,
        #                                    noisy_action2, self._action_spec)

        # action = action_target1

        # double Q-learning (use action instead of exp.action)
        # mqv2, critic2_state = self._critic_network2(
        #     (exp.observation, exp.action), state=state.critic2)

        # constructing the dual loss

        state = NafCriticState(
            critic1=critic1_state,
            critic2=critic2_state,
            target_actor=target_actor_state,
            target_critic1=target_critic1_state,
            target_critic2=target_critic2_state)

        info = NafCriticInfo(
            q_value1=q_value1,
            q_value2=q_value2,
            target_q_value1=target_q_value1,
            target_q_value2=target_q_value2)

        return state, info

    def _actor_train_step(self, exp: Experience, state: NafActorState):

        action, actor_state = self._actor_network(
            exp.observation, state=state.actor)

        q_value1, critic_state1 = self._critic_network1(
            (exp.observation, action), state=state.critic1)

        q_value2, critic_state2 = self._critic_network2(
            (exp.observation, action), state=state.critic2)
        q_value = torch.min(q_value1, q_value2)

        dqda = nest.pack_sequence_as(
            action,
            list(torch.autograd.grad(q_value.sum(), nest.flatten(action))))

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            loss = loss.sum(list(range(1, loss.ndim)))
            return loss

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        state = NafActorState(
            actor=actor_state, critic1=critic_state1, critic2=critic_state2)
        info = LossInfo(loss=sum(nest.flatten(actor_loss)), extra=actor_loss)
        return AlgStep(output=action, state=state, info=info)

    def _get_q_value(self, inputs, state=None):

        mqv1, critic_state1 = self._critic_network1(inputs, state=state)
        #mqv2, critic_state = self._critic_network2(inputs, state=state)

        q_value = mqv1[1].view(-1)

        return q_value, critic_state1

    def _get_state_value(self, inputs, state=None):

        mqv1, critic_state1 = self._critic_network1(inputs, state=state)

        state_value = mqv1[2].view(-1)

        return state_value, critic_state1

    def train_step(self, exp: Experience, state: NafState):
        critic_state, critic_info = self._critic_train_step(
            exp=exp, state=state.critic)
        policy_step = self._actor_train_step(exp=exp, state=state.actor)

        return policy_step._replace(
            state=NafState(actor=policy_step.state, critic=critic_state),
            info=NafInfo(
                action_distribution=policy_step.output,
                critic=critic_info,
                actor_loss=policy_step.info))

    def calc_loss(self, training_info: TrainingInfo):

        critic_loss1 = self._critic_loss1(
            training_info=training_info,
            value=training_info.info.critic.q_value1,
            target_value=training_info.info.critic.target_q_value1)
        critic_loss2 = self._critic_loss2(
            training_info=training_info,
            value=training_info.info.critic.q_value2,
            target_value=training_info.info.critic.target_q_value2)

        actor_loss = training_info.info.actor_loss

        return LossInfo(
            loss=critic_loss1.loss + critic_loss2.loss + actor_loss.loss,
            extra=NafLossInfo(
                critic1=critic_loss1.extra,
                critic2=critic_loss2.extra,
                actor=actor_loss))

    def after_update(self, training_info):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return [
            '_target_actor_network', '_target_critic_network1',
            '_target_critic_network2'
        ]
