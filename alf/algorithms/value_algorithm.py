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
"""Value Algorithm."""

import numpy as np
import gin

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
from alf.networks import CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils

SacCriticState = namedtuple(
    "SacCriticState",
    ["critic1", "critic2", "target_critic1", "target_critic2"])

SacState = namedtuple("SacState", ["critic"])

SacCriticInfo = namedtuple("SacCriticInfo",
                           ["critic1", "critic2", "target_critic"])

SacInfo = namedtuple("SacInfo", ["critic"], default_value=())

SacLossInfo = namedtuple('SacLossInfo', ('critic'))


@gin.configurable
class ValueAlgorithm(OffPolicyAlgorithm):
    """Value Algorithm
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 critic_network: CriticNetwork,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 critic_optimizer=None,
                 debug_summaries=False,
                 name="SacAlgorithm"):
        """Create a SacAlgorithm

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network (Network): The network will be called with
                call(observation).
            critic_network (Network): The network will be called with
                call(observation, action).
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            critic_loss (None|OneStepTDLoss): an object for calculating critic loss.
                If None, a default OneStepTDLoss will be used.
            initial_log_alpha (float): initial value for variable log_alpha
            target_entropy (float|None): The target average policy entropy, for updating alpha.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            gradient_clipping (float): Norm length to clip gradients.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        critic_network1 = critic_network.copy()
        critic_network2 = critic_network.copy()

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=SacState(
                critic=SacCriticState(
                    critic1=critic_network.state_spec,
                    critic2=critic_network.state_spec,
                    target_critic1=critic_network.state_spec,
                    target_critic2=critic_network.state_spec)),
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)
        self.add_optimizer(critic_optimizer,
                           [critic_network1, critic_network2])

        self._critic_network1 = critic_network1
        self._critic_network2 = critic_network2
        self._target_critic_network1 = self._critic_network1.copy()
        self._target_critic_network2 = self._critic_network2.copy()

        if critic_loss is None:
            critic_loss = OneStepTDLoss(debug_summaries=debug_summaries)
        self._critic_loss = critic_loss

        flat_action_spec = nest.flatten(self._action_spec)
        self._flat_action_spec = flat_action_spec

        self._is_continuous = flat_action_spec[0].is_continuous

        self._update_target = common.get_target_updater(
            models=[self._critic_network1, self._critic_network2],
            target_models=[
                self._target_critic_network1, self._target_critic_network2
            ],
            tau=target_update_tau,
            period=target_update_period)

    def _critic_train_step(self, exp: Experience, state: SacCriticState,
                           action, log_pi):
        if self._is_continuous:
            critic_input = (exp.observation, exp.action)
            target_critic_input = (exp.observation, action)
        else:
            critic_input = exp.observation
            target_critic_input = exp.observation

        critic1, critic1_state = self._critic_network1(
            critic_input, state=state.critic1)

        critic2, critic2_state = self._critic_network2(
            critic_input, state=state.critic2)

        target_critic1, target_critic1_state = self._target_critic_network1(
            target_critic_input, state=state.target_critic1)

        target_critic2, target_critic2_state = self._target_critic_network2(
            target_critic_input, state=state.target_critic2)

        if not self._is_continuous:
            exp_action = exp.action.view(critic1.shape[0], -1).long()
            critic1 = critic1.gather(-1, exp_action)
            critic2 = critic2.gather(-1, exp_action)
            sampled_action = action.view(critic1.shape[0], -1).long()
            target_critic1 = target_critic1.gather(-1, sampled_action)
            target_critic2 = target_critic2.gather(-1, sampled_action)

        target_critic = torch.min(target_critic1, \
                                  target_critic2).reshape(log_pi.shape) - \
                         (torch.exp(self._log_alpha) * log_pi).detach()

        critic1 = critic1.squeeze(-1)
        critic2 = critic2.squeeze(-1)
        target_critic = target_critic.squeeze(-1).detach()

        state = SacCriticState(
            critic1=critic1_state,
            critic2=critic2_state,
            target_critic1=target_critic1_state,
            target_critic2=target_critic2_state)

        info = SacCriticInfo(
            critic1=critic1, critic2=critic2, target_critic=target_critic)

        return state, info

    def train_step(self, exp: Experience, state: SacState):

        critic_state, critic_info = self._critic_train_step(
            exp, state.critic, action, log_pi)

        state = SacState(
            share=SacShareState(actor=share_actor_state),
            actor=actor_state,
            critic=critic_state)
        info = SacInfo(
            action_distribution=action_distribution,
            actor=actor_info,
            critic=critic_info,
            alpha=alpha_info)
        return AlgStep(action, state, info)

    def after_update(self, training_info):
        self._update_target()

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss = self._calc_critic_loss(training_info)
        alpha_loss = training_info.info.alpha.loss
        actor_loss = training_info.info.actor.loss
        return LossInfo(
            loss=actor_loss.loss + critic_loss.loss + alpha_loss.loss,
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss.extra))

    def _calc_critic_loss(self, training_info):
        critic_info = training_info.info.critic

        target_critic = critic_info.target_critic

        critic_loss1 = self._critic_loss(
            training_info=training_info,
            value=critic_info.critic1,
            target_value=target_critic)

        critic_loss2 = self._critic_loss(
            training_info=training_info,
            value=critic_info.critic2,
            target_value=target_critic)

        critic_loss = critic_loss1.loss + critic_loss2.loss
        return LossInfo(loss=critic_loss, extra=critic_loss)

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_network1', '_target_critic_network2']
