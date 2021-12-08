"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

def _test_partial_sum_to_broadcast(test_case, src_device_type, dst_device_type):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    def build_p2b(input_blob, src_device_num, dst_device_num):
        with flow.scope.placement(src_device_type, "0:0-" + str(src_device_num - 1)):
            src = flow.identity(input_blob.with_distribute(flow.distribute.split(0)))
            src = flow.math.reduce_sum(src, axis=0)
        with flow.scope.placement(dst_device_type, "0:0-" + str(dst_device_num - 1)):
            dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
        return dst

    @flow.global_function(function_config=func_config)
    def partial_sum_to_broadcast_job(input_blob: oft.Numpy.Placeholder((96, 96, 96))):
        result_list = []
        for i in (2, 3):
            for j in (1, 2, 3):
                result_list.append(build_p2b(input_blob, i, j))
        return tuple(result_list)

    x = np.random.uniform(-1e-05, 1e-05, (96, 96, 96)).astype(np.float32)
    result_tuple = partial_sum_to_broadcast_job(x).get()
    for out in result_tuple:
        test_case.assertTrue(np.allclose(np.sum(x, axis=0), out.numpy()))

@flow.unittest.skip_unless_1n4d()
class TestBoxingV2(flow.unittest.TestCase):

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_partial_sum_to_broadcast(test_case):
        arg_dict = OrderedDict()
        arg_dict["src_device_type"] = ["cpu", "gpu"]
        arg_dict["dst_device_type"] = ["cpu", "gpu"]
        for arg in GenArgList(arg_dict):
            _test_partial_sum_to_broadcast(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
