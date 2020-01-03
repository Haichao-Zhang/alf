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
r"""Train model.

To run actor_critic on gym CartPole:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.train \
  --root_dir=~/tmp/cart_pole \
  --gin_file=ac_cart_pole.gin \
  --gin_param='create_environment.num_parallel_environments=8' \
  --alsologtostderr
```

You can view various training curves using Tensorboard by running the follwoing
command in a different terminal:
```bash
tensorboard --logdir=~/tmp/cart_pole
```

You can visualize playing of the trained model by running:
```bash
cd ${PROJECT}/alf/examples;
python -m alf.bin.play \
  --root_dir=~/tmp/cart_pole \
  --gin_file=ac_cart_pole.gin \
  --alsologtostderr
```

"""

import os

from absl import app
from absl import flags
from absl import logging

import gin.tf.external_configurables
from alf.utils import common
import alf.utils.external_configurables
from alf.trainers import policy_trainer

from alf.trainers import on_policy_trainer
from alf.trainers import off_policy_trainer
from alf.algorithms import agent
from alf.algorithms import multi_agent_algorithm
from alf.algorithms import reward_estimation
from alf.algorithms import ppo_algorithm

from alf.utils import multi_modal_actor_distribution_network
from tf_agents.networks.encoding_network import EncodingNetwork

from alf.environments.utils import create_environment
from alf.utils import common

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

# def get_trainer_config(root_dir):
#     trainer_config = policy_trainer.TrainerConfig(
#         root_dir=root_dir,
#         trainer=on_policy_trainer.OnPolicyTrainer,  # no (), only class name
#         unroll_length=8,
#         algorithm_ctor=agent.Agent,  # no (), only class name
#         num_iterations=20,
#         checkpoint_interval=20,
#         evaluate=True,
#         eval_interval=5,
#         debug_summaries=False,
#         summarize_grads_and_vars=False,
#         summary_interval=5)

#     return trainer_config


def get_m_algo():
    with gin.config_scope('reward'):
        feature_net = EncodingNetwork()
    print("----feature net==============")
    print(feature_net)

    print('featurenet before reward estimator')
    print(feature_net)
    # as separate fixed network
    with gin.config_scope('reward'):
        feature_net2 = EncodingNetwork()

    reward_estimator = reward_estimation.RewardAlgorithmState(
        fuse_net=feature_net)

    # print("para value after creation-----=====================-----")
    # print(feature_net.variables)
    with gin.config_scope('learner'):
        learner_network = multi_modal_actor_distribution_network.MultiModalActorDistributionNetworkMapping(
            feature_mapping=feature_net)
        learner_ppo_algo = ppo_algorithm.PPOAlgorithm(
            actor_network=learner_network)
    with gin.config_scope('teacher'):
        teacher_ppo_algo = ppo_algorithm.PPOAlgorithm()

    # print("------reward estimator==================")
    # print(reward_estimator._fuse_net.variables)
    # print("------ppo*******************************")
    # print(learner_ppo_algo._actor_network._feature_mapping.variables)

    m_alg = multi_agent_algorithm.MultiAgentAlgorithm(
        intrinsic_curiosity_module=reward_estimator,
        algos=[learner_ppo_algo, teacher_ppo_algo],
        debug_summaries=True)

    return m_alg


# ------------------------------------------------------------------
# for imitation learning ==============================
def get_trainer_config(root_dir):
    # multi-agent algorithm
    # use instances instead of constructors

    # feature net will be shared between policy (thus algorithm) and reward estimation

    # print("-------algorithm--------")
    # print(m_alg.__dict__)

    # print("========================================")
    # print(reward_estimator._fuse_net.variables)
    # print('----------------------------------------')
    # print(m_alg._algos[0]._actor_network._feature_mapping.variables)

    # trainer_config = policy_trainer.TrainerConfig(
    #     root_dir=root_dir,
    #     trainer=off_policy_trainer.
    #     SyncOffPolicyTrainer,  # no (), only class name
    #     algorithm_ctor=multi_agent_algorithm.
    #     MultiAgentAlgorithm  # no (), only class name)
    # )

    trainer_config = policy_trainer.TrainerConfig(
        root_dir=root_dir,
        algorithm_ctor=get_m_algo()  # no (), only class name)
    )

    return trainer_config


@gin.configurable
def train_eval(root_dir):
    """Train and evaluate algorithm

    Args:
        root_dir (str): directory for saving summary and checkpoints
    """

    #trainer_conf = policy_trainer.TrainerConfig(root_dir=root_dir)
    trainer_conf = get_trainer_config(root_dir)
    print('------before-------')
    # global _env
    # print(_env)
    # del _env

    trainer = trainer_conf.create_trainer()
    trainer.initialize()
    trainer.train()


def main(_):
    gin_file = common.get_gin_file()
    # still need to parse
    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_param)
    # python file
    # create the environment first
    # env = create_environment(nonparallel=True)
    env = create_environment(num_parallel_environments=1, nonparallel=False)
    common.set_global_env(env)
    # #global _env
    # print(env)
    # del env

    train_eval(FLAGS.root_dir)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    flags.mark_flag_as_required('root_dir')
    app.run(main)
