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

import tensorflow as tf
from collections import namedtuple, OrderedDict
import numpy as np


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(
            units=4,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer())

        self.dense2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(),
            bias_initializer=tf.random_normal_initializer())

    def call(self, input):
        output = self.dense2(self.dense1(input))
        return output

    def makeCheckpoint(self):
        return tf.train.Checkpoint(dense1=self.dense1)

    # def restoreVariables(self, path):
    #     status = self.ckpt.restore(tf.train.latest_checkpoint(path))
    #     status.assert_consumed()  # Optional check


X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

model = Linear()
y_pred = model(X)
print(model.summary())
print(model.variables)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# for i in range(1):
#     with tf.GradientTape() as tape:
#         y_pred = model(X)
#         print(y_pred)
#         loss = tf.reduce_mean(tf.square(y_pred - y))
#         print(loss)
#     grads = tape.gradient(loss, model.variables)

#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# print('---after sgd')
# print(model.variables)

# save_path_with_prefix = './checkpoints/ck'
# #checkpoint = tf.train.Checkpoint(model=model)
# checkpoint = model.makeCheckpoint()
# # saving
# checkpoint.save(save_path_with_prefix)
checkpoint = model.makeCheckpoint()
latest = tf.train.latest_checkpoint('./checkpoints/')
checkpoint.restore(latest)

print(latest)

# ##=================load
# ckpt = tf.train.get_checkpoint_state('./checkpoints/')
# pretrained_model = ckpt.model_checkpoint_path
# print(pretrained_model)

print(model.summary())
print(model.variables)

global env
env = "global_env"

print(env)


def fun():
    global env
    print(env)
    print('-----after print-------')

    del env


fun()
print("------after delete----")
print(env)
