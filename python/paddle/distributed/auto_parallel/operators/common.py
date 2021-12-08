# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

_g_distributed_operator_impl_registries = {}


class DistributedOperatorImplContainer:
    def __init__(self):
        self._impls = []
        self._name = None

    def register_impl(self, dist_impl):
        self._impls.append(dist_impl)

    def get_impl(self, impl_idx):
        return self._impls[impl_idx]

    def get_impls(self):
        return self._impls


class DistributedOperatorImpl:
    def __init__(self):
        self._name = None
        self._forward_implemented = False
        self._backward_implemented = False

    @staticmethod
    def forward(dist_ctx, *args, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @staticmethod
    def backward(dist_ctx, *grad_outputs, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def get_name(self):
        return self._name

    def is_input_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def is_output_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def is_compatible(self, dist_op):
        return self.is_input_compatible(dist_op) and \
            self.is_output_compatible(dist_op)

    def update_dims_mapping(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")


def register_distributed_operator_impl_container(name, dist_op_impl_container):
    global _g_distributed_operator_impl_registries
    _g_distributed_operator_impl_registries[name] = dist_op_impl_container


def get_distributed_operator_impl_container(name):
    global _g_distributed_operator_impl_registries
    return _g_distributed_operator_impl_registries.get(name, None)


def register_distributed_operator_impl(name, dist_impl):
    dist_op_impl_container = get_distributed_operator_impl_container(name)
    if dist_op_impl_container is not None:
        dist_op_impl_container.register_impl(dist_impl)
    else:
        assert False, "Must register distributed operator registry first."


def get_distributed_operator_impl(name, impl_idx):
    global _g_distributed_operator_impl_registries
    return _g_distributed_operator_impl_registries[name].get_impl(impl_idx)


def find_best_compatible_distributed_operator_impl(name, dist_op, fwd=True):
    """
    Here just return the first compatible implemention. 
    This will be improved by cost model in the future.
    """
    dist_op_impl_container = get_distributed_operator_impl_container(name)
    if dist_op_impl_container is None:
        return None, -1
    compatible_impls = []
    impls = dist_op_impl_container.get_impls()
    if fwd:
        for idx, impl in enumerate(impls):
            if impl.is_input_compatible(dist_op):
                compatible_impls.append((impl, idx))
    else:
        for idx, impl in enumerate(impls):
            if impl.is_output_compatible(dist_op):
                compatible_impls.append((impl, idx))

    if compatible_impls:
        best_compatible_impl, idx = compatible_impls[0]
    else:
        best_compatible_impl, idx = None, -1

    return best_compatible_impl, idx


def infer_shape(block, src_var, src_var_dist_attr, op_input_dist_attr):
    var_shape = block.var(src_var.name).shape
    var_topoloy = src_var_dist_attr.process_mesh.topology
    var_dims_mapping = src_var_dist_attr.dims_mapping

    complete_shape = []
    for idx, shape in enumerate(var_shape):
        if var_dims_mapping[idx] == -1:
            complete_shape.append(shape)
        else:
            new_shape = shape * var_topoloy[var_dims_mapping[idx]]
            complete_shape.append(new_shape)

    exact_shape = []
    input_topology = op_input_dist_attr.process_mesh.topology
    input_dims_mapping = op_input_dist_attr.dims_mapping
    for idx, shape in enumerate(complete_shape):
        if input_dims_mapping[idx] == -1:
            exact_shape.append(shape)
        else:
            new_shape = shape // input_topology[input_dims_mapping[idx]]
            exact_shape.append(new_shape)

    return exact_shape