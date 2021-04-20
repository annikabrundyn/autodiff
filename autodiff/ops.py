import re
import numpy as np
import numbers
from .node import Node, Variable, add_context
from .reshape import ReduceSumKeepDims

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
# def letters_from_tuple(tpl):
#     return ascii_lowercase[:len(tpl)]
# 
# 
# def shape_from_elems(*elems):
#     if len(elems) == 0:
#         return 1,
#     return np.broadcast(*[np.ones(elem.shape) for elem in elems]).shape


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
        return ReduceSumToShape(grad, wrt.shape)


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
        return ReduceSumToShape(grad, wrt.shape)


class Negate(Node):
    def __init__(self, node, name="Negate"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return -self.node()

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return -previous_grad
        else:
            return 0


class Recipr(Node):
    def __init__(self, node, name="Reciprocal"):
        """
        Elementwise reciprocal

        """
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return 1 / (self.node() + Node.epsilon)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return - previous_grad * self * self
        return 0


# class Einsum(Node):
#     def __init__(self, op_str, *operands, name="EinSum"):
#         super().__init__(list(operands), name + " " + op_str)
#         # TODO ellipsis currently can't be in the middle of op_letters!
#         self.op_str = op_str
#         self.operands = self.children
# 
#         self.opnames = re.split(",|->", self.op_str)
#         self.all_letters = "".join(set("".join(self.opnames[:-1])))
#         # can also be "..." to an arbitrary shape tuple
#         self.letter_to_dim = {}
# 
#         if len(self.operands) + 1 != len(self.opnames):
#             raise ValueError("Number of operands doesn't match the einsum string!")
# 
#         for op, op_letters in zip(self.operands, self.opnames[:-1]):
#             if len(op.shape) != 0 and len(op.shape) != len(op_letters) \
#                     and "..." not in op_letters and op_letters != "":
#                 raise ValueError("Dimension of operand " + str(op) + " doesn't match the string! " +
#                                  "Shape: " + str(op.shape) + " , string: '" + op_letters + "'")
# 
#             shp = op.shape
#             if op_letters[:3] == "...":
#                 op_letters = op_letters[::-1]
#                 shp = op.shape[::-1]
#             for i, lett in enumerate(Einsum.split_dots(op_letters)):
#                 try:
#                     if len(lett) == 1:
#                         dim = [shp[i]]  # what if shape is an empty tuple?
#                     else:
#                         dim = shp[i:]
#                     if self.letter_to_dim.get(lett, dim) != dim:
#                         raise ValueError("Inconsistent dimension names!")
#                     self.letter_to_dim[lett] = dim
#                 except IndexError:
#                     pass  # letters that we can't add are just dimension 1
# 
#         self.shape = []
#         for let in Einsum.split_dots(self.opnames[-1]):
#             for l in self.letter_to_dim.get(let, [1]):
#                 self.shape.append(l)
#         self.shape = tuple(self.shape)
# 
#     @staticmethod
#     def split_dots(op_str):
#         match_string = "\.{3}|\S"
#         return re.findall(match_string, op_str)
# 
#     def _eval(self):
#         arr = [op() for op in self.operands]
# 
#         for i, val in enumerate(arr):
#             if isinstance(val, numbers.Number):
#                 shp = [l for let in Einsum.split_dots(self.opnames[i]) for l in self.letter_to_dim.get(let, [1])]
#                 arr[i] = np.broadcast_to(val, shp)
# 
#         return np.einsum(self.op_str, *arr)
# 
#     def _partial_derivative(self, wrt, previous_grad):
#         """
#         Usual einsum operation looks something like this c = einsum("ij,jk->ik", a, b)
#         Gradient w.r.t. the first parameter just changes the op to look like this: df = einsum("ik,jk->ij", c, b).
#         It basically just switches the output with one of the inputs.
# 
#         For tensors that have some of their dimensions implicitly summed, a new tensor of ones is explicitly added
#         """
#         order = list(range(len(self.opnames)))
# 
#         try:
#             loc = self.operands.index(wrt)
#         except ValueError:
#             return 0
#         order[loc], order[-1] = order[-1], order[loc]
# 
#         # this is concatenation of two lists in np array and then their reorder
#         operands_with_grad = list(np.array(self.operands + [previous_grad])[order])
# 
#         opnames = list(np.array(self.opnames)[order])
# 
#         # here we add explicit Variables for implicitly summed out tensors
#         for i, letter in enumerate(Einsum.split_dots(self.opnames[loc])):
#             if letter not in Einsum.split_dots("".join(opnames[:-1])):
#                 opnames.insert(0, letter)
# 
#                 dim = wrt.shape[i]
#                 var_to_insert = Variable(np.ones(dim), name="np.ones(" + str(dim) + ")")
#                 operands_with_grad.insert(0, var_to_insert)
#         op_str = Einsum.to_einsum_string(opnames)
# 
#         return Einsum(op_str, *operands_with_grad[:-1])
# 
#     @staticmethod
#     def to_einsum_string(list_of_ops):
#         return ",".join(list_of_ops[:-1]) + "->" + list_of_ops[-1]





class Pow(Node):
    def __init__(self, first, second, name="Pow"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]
        self.shape = shape_from_elems(*self.children)

    def _eval(self):
        return np.power(self.first(), self.second())

    def _partial_derivative(self, wrt, previous_grad):
        if self.first == self.second == wrt:
            return previous_grad * self * (Log(self.first) + 1)
        elif self.first == wrt:
            return previous_grad * self.second * Pow(self.first, self.second - 1)
        elif self.second == wrt:
            return previous_grad * Log(self.first) * self
        return 0


class Log(Node):
    def __init__(self, node, name="Log"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.log(self.node() + Node.epsilon)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * Recipr(self.node)
        return 0


class Identity(Node):
    def __init__(self, node, name="Identity"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return self.node()

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad
        return 0


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
