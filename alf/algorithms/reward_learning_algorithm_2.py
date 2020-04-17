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

from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple, TimeStep
from alf.nest import nest
from alf.nest.utils import NestConcat
from alf.networks import Network, EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import losses, math_ops, spec_utils, tensor_utils

RewardState = namedtuple(
    "RewardState", ["feature", "network"], default_value=())
RewardInfo = namedtuple("RewardInfo", ["loss"])


@gin.configurable
class RewardEstimationAlgorithm(Algorithm):
    """Base Dynamics Learning Module

    This module trys to learn the dynamics of environment.
    """

    def __init__(self,
                 train_state_spec,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 dynamics_network: Network = None,
                 name="RewardEstimationAlgorithm"):
        """Create a RewardEstimationAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            dynamics_network (Network): network for predicting next feature
                based on the previous feature and action. It should accept
                input with spec [feature_spec, encoded_action_spec] and output
                a tensor of shape feature_spec. For discrete action,
                encoded_action is an one-hot representation of the action.
                For continuous action, encoded action is the original action.
        """
        super().__init__(train_state_spec=train_state_spec, name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        if action_spec.is_discrete:
            self._num_actions = action_spec.maximum - action_spec.minimum + 1
        else:
            self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._feature_spec = feature_spec

        #feature_dim = flat_feature_spec[0].shape[-1]

        if isinstance(hidden_size, int):
            hidden_size = (hidden_size, )

        if dynamics_network is None:
            encoded_action_spec = TensorSpec((self._num_actions, ),
                                             dtype=torch.float32)
            dynamics_network = EncodingNetwork(
                name="dynamics_net",
                input_tensor_spec=(feature_spec, encoded_action_spec,
                                   feature_spec),
                preprocessing_combiner=NestConcat(),
                activation=torch.relu,
                fc_layer_params=hidden_size,
                last_layer_size=1,
                last_activation=math_ops.identity)

        self._dynamics_network = dynamics_network

    def _encode_action(self, action):
        if self._action_spec.is_discrete:
            return torch.nn.functional.one_hot(
                action, num_classes=self._num_actions)
        else:
            return action

    def update_state(self, time_step: TimeStep, state: RewardState):
        """Update the state based on TimeStep data. This function is
            mainly used during rollout together with a planner
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for DML (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: feature
                info (DynamicsInfo):
        """
        pass

    def get_state_specs(self):
        """Get the state specs of the current module.
        This function is mainly used for constructing the nested state specs
        by the upper-level module.
        """
        pass

    def train_step(self, time_step: TimeStep, state: RewardState):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (RewardState): state for training
                info (RewardInfo):
        """
        pass

    def calc_loss(self, info: RewardInfo):
        loss = nest.map_structure(torch.mean, info.loss)
        return LossInfo(
            loss=info.loss, scalar_loss=loss.loss, extra=loss.extra)


@gin.configurable
class RewardAlgorithm(RewardEstimationAlgorithm):
    """Deterministic Dynamics Learning Module

    This module trys to learn the dynamics of environment with a
    determinstic model.
    """

    def __init__(self,
                 action_spec,
                 feature_spec,
                 hidden_size=256,
                 dynamics_network: Network = None,
                 name="RewardAlgorithm"):
        """Create a RewardAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            dynamics_network (Network): network for predicting next feature
                based on the previous feature and action. It should accept
                input with spec [feature_spec, encoded_action_spec] and output
                a tensor of shape feature_spec. For discrete action,
                encoded_action is an one-hot representation of the action.
                For continuous action, encoded action is the original action.
        """
        if dynamics_network is not None:
            dynamics_network_state_spec = dynamics_network.state_spec
        else:
            dynamics_network_state_spec = ()

        reward_spec = TensorSpec((1, ), dtype=torch.float32)

        super().__init__(
            train_state_spec=RewardState(
                feature=feature_spec, network=dynamics_network_state_spec),
            action_spec=action_spec,
            feature_spec=feature_spec,
            hidden_size=hidden_size,
            dynamics_network=dynamics_network,
            name=name)

    def predict_step(self, time_step: TimeStep, state: RewardState):
        """Predict the next observation given the current time_step.
                The next step is predicted using the prev_action from time_step
                and the feature from state.
        """
        prev_action = self._encode_action(time_step.prev_action)
        prev_obs = state.feature
        current_obs = time_step.observation
        forward_pred, network_state = self._dynamics_network(
            inputs=(prev_obs, prev_action, current_obs), state=state.network)

        # forward_pred = spec_utils.scale_to_spec(forward_pred.tanh(),
        #                                         self._feature_spec)
        # state = state._replace(feature=forward_pred, network=network_state)
        # pass the observation to next
        state = state._replace(
            feature=time_step.observation, network=network_state)
        return AlgStep(output=forward_pred, state=state, info=())

    def update_state(self, time_step: TimeStep, state: RewardState):
        """Update the state based on TimeStep data. This function is
            mainly used during rollout together with a planner
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for DML (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: feature
                info (RewardInfo):
        """
        feature = time_step.observation
        return state._replace(feature=feature)

    def train_step(self, time_step: TimeStep, state: RewardState):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (RewardState): state for training
                info (RewardInfo):
        """
        feature = time_step.reward
        dynamics_step = self.predict_step(time_step, state)
        forward_pred = dynamics_step.output
        # forward_loss = losses.element_wise_squared_loss(feature, forward_pred)
        # BUG: here fea should be [B, 1], not [B]
        forward_loss = (feature.view(-1, 1) - forward_pred)**2
        forward_loss = 0.5 * forward_loss.mean(
            list(range(1, forward_loss.ndim)))
        info = RewardInfo(
            loss=LossInfo(
                loss=forward_loss, extra=dict(reward_loss=forward_loss)))
        # pass observation down
        state = RewardState(feature=time_step.observation)

        return AlgStep(output=(), state=state, info=info)

    def compute_reward(self, obs, action, obs_current, state=None):
        """Predict the next observation given the current time_step.
                The next step is predicted using the prev_action from time_step
                and the feature from state.
        """
        action = self._encode_action(action)
        forward_pred, _ = self._dynamics_network(
            inputs=(obs, action, obs_current), state=())
        return forward_pred
