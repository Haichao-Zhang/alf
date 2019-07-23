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

import os
import functools
import tensorflow as tf
import numpy as np

FLOAT32_MAX = np.finfo(np.float32).max


def timer(
        record=os.environ.get('TIMER_RECORD', False),
        interval=int(os.environ.get('TIMER_INTERVAL', 1000)),
        name=None):
    """Decorator to record time cost for a function

    NOTE: It's just a helper function for time profiling, it's not thread safe

    Args:
        record (bool): A bool whether to record time cost for function, we can specify
            it explicitly or config through environment variable `TIMER_RECORD`. It's used
            as a tool function for profiling, do not set it to `True` only if in debugging
        interval (int): summary time cost every so many interval
        name (str): name for this timer
    """

    def wrapper(func):
        if not record:
            return func

        # timer for different instances
        timers = {}

        @functools.wraps(func)
        def _wrapper(self, *args, **kwargs):
            if self not in timers:
                timers[self] = Timer(func=func, interval=interval, name=name)

            return timers[self](self, *args, **kwargs)

        return _wrapper

    return wrapper


def _get_func_qualname(func):
    return getattr(func, 'python_function', func).__qualname__


class Timer(object):
    def __init__(self, func, interval, name=None):
        """Timer that record time cost for every function call and summary periodically
        Args:
            func (Callable): function to record
            interval (int): summary time cost every so many interval
            name (str): name for this timer
        """

        self._interval = interval
        self._func = func
        self._name = _get_func_qualname(func) if name is None else name
        self._counter = tf.Variable(
            initial_value=0, dtype=tf.int64, trainable=False)
        self._total = tf.Variable(
            initial_value=0, dtype=tf.float64, trainable=False)
        self._max = tf.Variable(
            initial_value=0, dtype=tf.float64, trainable=False)
        self._min = tf.Variable(
            initial_value=0, dtype=tf.float64, trainable=False)
        self._reset()

    def __call__(self, *args, **kwargs):
        start_time = tf.timestamp()
        ret_value = self._func(*args, **kwargs)
        duration = (tf.timestamp() - start_time) * 1e3
        self._counter.assign_add(1)
        self._min.assign(tf.math.minimum(self._min, duration))
        self._max.assign(tf.math.maximum(self._max, duration))
        self._total.assign_add(duration)

        tf.cond(
            tf.equal(tf.math.mod(self._counter, self._interval), 0),
            self._summary, lambda: tf.constant(False))

        return ret_value

    def _summary(self):
        prefix = 'time_cost/{}/Latest_{}_'.format(self._name, self._interval)
        with tf.summary.record_if(True):
            if self._interval == 1:
                tf.summary.scalar(
                    name='time_cost/{}'.format(self._name),
                    data=self._total,
                    step=self._counter)
            else:
                tf.summary.scalar(
                    name=prefix + 'avg',
                    data=self._total / tf.cast(
                        self._interval, dtype=tf.float64),
                    step=self._counter)
                tf.summary.scalar(
                    name=prefix + 'min', data=self._min, step=self._counter)
                tf.summary.scalar(
                    name=prefix + 'max', data=self._max, step=self._counter)
        self._reset()
        return tf.constant(True)

    def _reset(self):
        self._total.assign(0)
        self._max.assign(0)
        self._min.assign(FLOAT32_MAX)
