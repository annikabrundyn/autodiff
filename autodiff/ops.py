import re
import numpy as np
import numbers
from node import Node, Variable

from functools import reduce
from string import ascii_lowercase


# def module_wrapper(fn):
#     def wrap_in_context(*args, **kwargs):
#         with add_context(fn.__name__):
#             return fn(*args, **kwargs)
# 
#     return wrap_in_context
# 
# 
def letters_from_tuple(tpl):
    return ascii_lowercase[:len(tpl)]


def shape_from_elems(*elems):
    if len(elems) == 0:
        return 1,
    return np.broadcast(*[np.ones(elem.shape) for elem in elems]).shape


# @module_wrapper
# def ReduceSumToShape(tensor, to_shape):
#     if tensor.shape == to_shape:
#         return tensor
#     previous_grad_letters = letters_from_tuple(tensor.shape)
#     if len(to_shape) == 0:
#         wrt_letters = ""
#     else:
#         wrt_letters = previous_grad_letters[-len(to_shape):]  # take last letters of previous_grad_letters
# 
#     new_curr_grad = Einsum(str(previous_grad_letters) + "->" + str(wrt_letters), tensor)
#     reduced_sum_grad = ReduceSumKeepDims(new_curr_grad, axes=[i for i, val in enumerate(to_shape) if val == 1])
#     return reduced_sum_grad


class Add(Node):
    def __init__(self, *elems, name="Add"):
        if not elems:
            name = "0-" + name
        super().__init__(list(elems), name)
        self.shape = shape_from_elems(*self.children)

    def _eval(self):
        # Using python sum instead of np.sum because python converts types correctly
        return np.array(sum([elem() for elem in self.children]))

    def _partial_derivative(self, wrt, previous_grad):
        # previous_grad will always be of shape of the shape of the "largest" variable
        # we need to sum across those other axes

        wrt_count = self.children.count(wrt)
        grad = previous_grad * Variable(wrt_count)
        return grad


class Mul(Node):
    fn = lambda x, y: x * y

    def __init__(self, *elems, name="Mul"):
        if not elems:
            name = "1-" + name
        super().__init__(list(elems), name)
        self.shape = shape_from_elems(*self.children)

    def _eval(self):
        # Mul broadcasts
        return reduce(Mul.fn, [child() for child in self.children], 1)

    def _partial_derivative(self, wrt, previous_grad):
        # previous_grad will always be of shape of the shape of the "largest" variable ?
        # we need to sum across those other axes ?
        add_list = []
        for loc, child in enumerate(self.children):
            if child == wrt:
                add_list.append(Mul(*[ch for i, ch in enumerate(self.children) if loc != i]))

        grad = previous_grad * Add(*add_list)
        return grad


class Exp(Node):
    def __init__(self, node, name="Exp"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.exp(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * self
        return 0
