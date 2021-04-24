import numbers
import numpy as np


class Node:

    def __init__(self, children, name="Node"):
        # wraps normal numbers into Variables
        self.children = [child if isinstance(child, Node) else Variable(child) for child in children]
        self.name = name
        self.cached = None

    def _eval(self):
        raise NotImplementedError()

    def _partial_derivative(self, wrt, previous_grad):
        raise NotImplementedError()

    def eval(self):
        # evaluate/calculate expression if not cached
        if self.cached is None:
            self.cached = self._eval()
        return self.cached

    def partial_derivative(self, wrt, previous_grad):
        return self._partial_derivative(wrt, previous_grad)

    def __call__(self, *args, **kwargs):
        return self.eval()

    def __str__(self):
        return self.name

    def __add__(self, other):
        from ops import Add
        return Add(self, other)

    def __neg__(self):
        from ops import Negate
        return Negate(self)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __mul__(self, other):
        from ops import Mul
        return Mul(self, other)
    

class Variable(Node):
    def __init__(self, value, name=None):
        if name is None:
            name = str(value)  # this op is really slow for np.arrays?!
        super().__init__([], name)

        if isinstance(value, numbers.Number):
            self._value = np.array(value, dtype=np.float64)
        else:
            self._value = value
        self.shape = self._value.shape

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self.cached = self._value = val

    def _eval(self):
        return self._value

    def _partial_derivative(self, wrt, previous_grad):
        if self == wrt:
            return previous_grad
        return 0