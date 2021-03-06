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
"""Unittests for nest_utils.py"""

from collections import namedtuple
import tensorflow as tf

from absl import logging

from alf.utils import nest_utils

NTuple = namedtuple('NTuple', ['a', 'b'])


class ListWrapper(list):
    pass


class TestListNest(tf.test.TestCase):
    def test_list_nest(self):
        list_nest = ('a', NTuple(a=1, b=2), (3, 4), list([5, 6]),
                     ListWrapper([7, 8]), dict(a=9, b=10))
        tuple_nest = nest_utils.nest_list_to_tuple(list_nest)
        expected_tuple_nest = ('a', NTuple(a=1, b=2), (3, 4), (5, 6), (7, 8),
                               dict(a=9, b=10))
        print(tuple_nest)
        self.assertEqual(tuple_nest, expected_tuple_nest)
        self.assertEqual(type(tuple_nest[1]), NTuple)
        self.assertFalse(nest_utils.nest_contains_list(tuple_nest))
        self.assertTrue(nest_utils.nest_contains_list(list_nest))
        new_list_nest = nest_utils.nest_tuple_to_list(tuple_nest, list_nest)
        print(new_list_nest)
        self.assertEqual(new_list_nest, list_nest)
        self.assertEqual(type(new_list_nest[1]), NTuple)
        self.assertEqual(type(new_list_nest[4]), ListWrapper)


class TestFindField(tf.test.TestCase):
    def test_find_field(self):
        nest = NTuple(a=1, b=NTuple(a=NTuple(a=2, b=3), b=2))
        ret = nest_utils.find_field(nest, 'a')
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0], nest.a)
        self.assertEqual(ret[1], nest.b.a)

        nest = (1, NTuple(a=NTuple(a=2, b=3), b=2))
        ret = nest_utils.find_field(nest, 'a')
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0], nest[1].a)

        nest = NTuple(a=1, b=[NTuple(a=2, b=3), 2])
        ret = nest_utils.find_field(nest, 'a')
        self.assertEqual(len(ret), 2)
        self.assertEqual(ret[0], nest.a)
        self.assertEqual(ret[1], nest.b[0].a)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf.test.main()
