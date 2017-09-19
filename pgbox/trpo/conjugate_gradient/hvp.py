import numpy as np
import tensorflow as tf
from pgbox.utils import *

def new_tensor(name, ndim, dtype):
    return tf.placeholder(dtype=dtype, shape=[None] * ndim, name=name)

def new_tensor_like(name, arr_like):
    return new_tensor(name, arr_like.get_shape().ndims, arr_like.dtype.base_dtype)

def flatten_tensor_variables(ts):
    return tf.concat(axis=0, values=[tf.reshape(x, [-1]) for x in ts])

def flat_to_tensors(flat, var_list):
    shapes = list(map(var_shape, var_list))
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = np.prod(shape)
        assigns.append(np.reshape(flat[start:start + size], shape))
        start += size
    return assigns

def build_fisher_hvp(f, var_list, cg_damping):
    params = var_list

    constraint_grads = tf.gradients(f, xs=params)
    for idx, (grad, param) in enumerate(zip(constraint_grads, params)):
        if grad is None:
            constraint_grads[idx] = tf.zeros_like(param)

    xs = tuple([new_tensor_like(p.name.split(":")[0], p) for p in params])

    Hx_plain_splits = tf.gradients(
        tf.reduce_sum(
            tf.stack([tf.reduce_sum(g * x) for g, x in zip(constraint_grads, xs)])
        ),
        params
    )
    for idx, (Hx, param) in enumerate(zip(Hx_plain_splits, params)):
        if Hx is None:
            Hx_plain_splits[idx] = tf.zeros_like(param)
    flatt = flatten_tensor_variables(Hx_plain_splits)


    def fisher_vector_product(p, session, feed_dict):
        ps = flat_to_tensors(p, var_list)
        for x, val in zipsame(xs, ps):
            feed_dict[x] = val
        return session.run(flatt, feed_dict) + p * cg_damping

    return fisher_vector_product
