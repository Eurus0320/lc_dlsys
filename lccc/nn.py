from . import ops


def softmax(node):
    exp_node = ops.exp_op(node)
    return exp_node / ops.reducesum_op(exp_node, reduction_indices=1, keepdims=True)


def softmax_cross_entropy_with_logits(logits, labels):
    h = softmax(logits)
    y = labels
    return -ops.reducesum_op(y * ops.log_op(h), reduction_indices=1)


def relu(node):
    return ops.relu_op(node)


def conv2d(input, filter, strides, padding):
    assert isinstance(strides, list) and len(strides) == 4
    assert strides[0] == 1 and strides[3] == 1
    assert padding == "SAME" or padding == "VALID"
    return ops.conv2d_op(input, filter, strides, padding)


def max_pool(value, ksize, strides, padding):
    assert isinstance(ksize, list) and len(ksize) == 4
    assert isinstance(strides, list) and len(strides) == 4
    assert ksize[0] == 1 and ksize[3] == 1
    assert strides[0] == 1 and strides[3] == 1
    assert padding == "SAME" or padding == "VALID"
    return ops.maxpool_op(value, ksize, strides, padding)


def dropout(input, keep_prob):
    return ops.dropout_op(input, keep_prob)