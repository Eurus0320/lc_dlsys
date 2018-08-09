from .base import  *
from . import ops
import numpy as np

variable_inits = []

def constant(init, shape):
    x = np.zeros(shape)
    x[:] = init
    return x

def placeholder(dtype = float32, shape = None):
    return ops.placeholder_op()

def Variable(init = None, dtype = float32):
    v = ops.variable_op()
    if init is not None:
        if not isinstance(init, np.ndarray):
            if not isinstance(init, list):
                init = [init]
            init = np.array(init)
        c = ops.const_op(init)
        variable_inits.append(ops.assign_op(v, c))
    return v


def sqrt(node):
    return ops.power_op(node, 0.5)

def power(node_a, node_b):
    return ops.power_op(node_a, node_b)


def log(node):
    return ops.log_op(node)


def matmul(node_a, node_b):
    return ops.matmul_op(node_a, node_b)

def reduce_sum(node, reduction_indices = None):
    if not isinstance(reduction_indices, list):
        reduction_indices = [0]
        assert len(reduction_indices) == 1
        return ops.reducesum_op(node, reduction_indices[0])

def reduce_mean(node, reduction_indices = None):
    return reduce_sum(node, reduction_indices) / ops.shape_op(node, reduction_indices)

def zeros(shape):
    return np.zeros(shape)

def equal(node_a, node_b):
    return ops.equal_op(node_a, node_b)

def argmax(node, axis = 0):
    return ops.argmax_op(node, axis)

def cast(node, dtype = float32):
    return node

def assign(assign_to, value):
    return ops.assign_op(assign_to, value)

def initialize_all_variables():
    global variable_inits
    init_node = ops.init_op(variable_inits)
    variable_inits = []
    return init_node

def global_variables_initializer():
    return initialize_all_variables()

def gradients(output_node, node_list):
    assert isinstance(output_node, ops.Node)
    assert isinstance(node_list, list)
    return ops.gradients(output_node, node_list)

def random_normal(shape, mean = 0.0, stddev = 1.0):
    return np.random.normal(loc = mean, scale = stddev, size = shape)

def reshape(node, shape):
    return ops.reshape_op(node, shape)