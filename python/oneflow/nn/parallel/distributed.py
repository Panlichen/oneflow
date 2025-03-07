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
import warnings
from collections import OrderedDict

import oneflow as flow
from oneflow.support.env_var_util import parse_boolean_from_env
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple


def grad_setting_fn(module, param):
    def grad_setting(grad):
        if param.grad is None:
            start = module._param_grad_offset_in_bucket[param]
            bucket_index = module._bucket_index[param]
            bucket_tensor = module._bucket_tensors[bucket_index]
            param.grad = flow._C.slice_view_1d_contiguous(
                bucket_tensor, start, start + param.numel()
            ).view(param.shape)
            param._is_grad_acc_inplace = True
        return grad

    return grad_setting


def allreduce_fn(module, param):
    ddp_state_for_reversed_params = module._ddp_state_for_reversed_params
    buckets = module._buckets
    bucket_tensors = module._bucket_tensors

    def allreduce(grad):
        ddp_state_for_reversed_params[param][0] = True
        for index, bucket in enumerate(buckets):
            deleted = all(ddp_state_for_reversed_params[x][1] for x in bucket)
            if deleted:
                continue

            assert not any(ddp_state_for_reversed_params[x][1] for x in bucket)

            all_params_in_bucket_ready = all(
                ddp_state_for_reversed_params[x][0] for x in bucket
            )
            if all_params_in_bucket_ready:
                for x in bucket:
                    ddp_state_for_reversed_params[x][1] = True
                # NOTE(jianhao)(higher-order-grad):
                # local allreduce doesn't have gradient function, higher-order grad may be unsupported
                flow._C.local_all_reduce(bucket_tensors[index], inplace=True)
            else:
                break

    return allreduce


def DistributedDataParallel(
    module: "flow.nn.Module",
    *,
    broadcast_buffers: bool = True,
    broadcast_parameters: bool = True,
    bucket_size: int = 10
):
    assert all(x.dtype == flow.float32 for x in module.parameters())
    if parse_boolean_from_env("ONEFLOW_DISABLE_VIEW", False):
        warnings.warn(
            "because the environment variable 'ONEFLOW_DISABLE_VIEW' is set to true, so the view mechanism is disabled, and we will set bucket_size = 1"
        )
        bucket_size = 1
    world_size = flow.env.get_world_size()
    if broadcast_parameters:
        with flow.no_grad():
            for x in module.parameters():
                requires_grad = x.requires_grad
                flow._C.comm_broadcast(x, inplace=True)
                # TODO: fix the bug that x's requires_grad is discarded
                # after flow._C.comm_broadcast
                x.requires_grad_(requires_grad)

    all_grad_size = sum([x.numel() for x in module.parameters()])
    if all_grad_size > 0:
        device = list(module.parameters())[0].device
        assert all(x.device == device for x in module.parameters())
    reversed_param_list = list(
        reversed(list([param for param in module.parameters() if param.requires_grad]))
    )
    module._param_grad_offset_in_bucket = {}

    def numel_in_bucket(tensor: flow.Tensor):
        def align(x: int, unit_size: int):
            return (x + (unit_size - 1)) // unit_size * unit_size

        # tensor memory should be align to 512 bytes for cuda operations,
        # 4 is the bytes of a float number
        # TODO(jianhao): expose the `kCudaMemAllocAlignSize` from C++ to
        # avoid this hardcoded "512"
        return align(tensor.numel(), 512 // 4)

    offset_in_bucket = 0
    with flow.no_grad():
        for i, param in enumerate(reversed_param_list):
            assert param.is_leaf
            if i % bucket_size == 0:
                offset_in_bucket = 0
            module._param_grad_offset_in_bucket[param] = offset_in_bucket
            offset_in_bucket += numel_in_bucket(param)

    module._bucket_index = {
        x: i // bucket_size for i, x in enumerate(reversed_param_list)
    }
    module._buckets = [
        reversed_param_list[i : i + bucket_size]
        for i in range(0, len(reversed_param_list), bucket_size)
    ]

    bucket_elems = 0
    module._bucket_tensors = []
    for b in module._buckets:
        bucket_elems = sum([numel_in_bucket(x) for x in b])
        module._bucket_tensors.append(
            flow.zeros(bucket_elems, dtype=flow.float32, device=device)
        )

    ddp_state_for_reversed_params = OrderedDict(
        reversed([(x, [False, False]) for x in module.parameters() if x.requires_grad])
    )
    module._ddp_state_for_reversed_params = ddp_state_for_reversed_params
    # The gradient shoule be averaged by all the nodes, so besides allreduce,
    # a division by world_size is required.
    # Use x * (1 / world_size) instead of x / world_size for two reasons:
    # 1. multiplication is faster than division
    # 2. An inplace operation is needed here (for allreduce grouping)
    #    But we do not have inplace division in oneflow.
    mul_factor = 1 / world_size

    def inplace_mul_and_return_none(x):
        x.mul_(mul_factor)
        return None

    for param in module.parameters():
        if param.requires_grad:
            param.register_hook(grad_setting_fn(module, param))
            param._register_post_grad_accumulation_hook(inplace_mul_and_return_none)
            param._register_post_grad_accumulation_hook(allreduce_fn(module, param))

    def post_forward_hook(module, input, output):
        ddp_state_for_reversed_params = module._ddp_state_for_reversed_params
        for state in ddp_state_for_reversed_params.values():
            state[0], state[1] = False, False
        if isinstance(output, (tuple, list)):
            if isinstance(output[0], dict):
                # For List[Dict[Tensor]] return type.
                out_key_list = []
                out_val_list = []
                for out in output:
                    out_keys = list(out.keys())
                    out_values = list(out.values())
                    out_key_list.append(out_keys)
                    out_val_list.extend(out_values)
                out_values = flow._C.select_top_n(
                    convert_to_tensor_tuple(
                        [*out_val_list, *ddp_state_for_reversed_params.keys()]
                    ),
                    n=len(out_val_list),
                )
                output = []
                for i, keys in enumerate(out_key_list):
                    output.append(
                        dict(zip(keys, out_values[i * len(keys) : (i + 1) * len(keys)]))
                    )
                return output
            else:
                # For List[Tensor] return type.
                output = flow._C.select_top_n(
                    convert_to_tensor_tuple(
                        [*output, *ddp_state_for_reversed_params.keys()]
                    ),
                    n=len(output),
                )
        elif isinstance(output, dict):
            # For Dict[Tensor] return type.
            out_keys = list(output.keys())
            out_values = list(output.values())
            out_values = flow._C.select_top_n(
                convert_to_tensor_tuple(
                    [*out_values, *ddp_state_for_reversed_params.keys()]
                ),
                n=len(out_values),
            )
            return dict(zip(out_keys, out_values))
        else:
            # For Tensor return type.
            output = flow._C.select_top_n(
                convert_to_tensor_tuple(
                    [output, *ddp_state_for_reversed_params.keys()]
                ),
                n=1,
            )[0]
        buffers = list(module.buffers())
        if len(buffers) > 0:
            flow._C.stream_touch(buffers)
        return output

    module.register_forward_hook(post_forward_hook)

    if broadcast_buffers:

        def pre_forward_hook(module, input):
            with flow.no_grad():
                buffers = list(module.buffers())
                flow._C.comm_broadcast(buffers, inplace=True)

        module.register_forward_pre_hook(pre_forward_hook)

    module._is_ddp_module = True

    return module
