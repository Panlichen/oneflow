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
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=2, check_graph=False)
def _test_einsum_eltwise_multiply(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ij,ij->ij", g_x, g_y)
    return z


class TestEinsumGlobal(flow.unittest.TestCase):
    @globaltest
    def test_einsum_eltwise_multiply(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_eltwise_multiply(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
