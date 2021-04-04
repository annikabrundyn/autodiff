from .operations import *


class Node(object):

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Subtract(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __pow__(self, power, modulo=None):
        return Power(self, power)

    def __floordiv__(self, other):
        return Divide(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class Variable(Node):
    def __init__(self, dtype, name=None):
        self.name = dtype if name is None else name
        self.dtype = dtype
        self.value = None
        self.a = None
        self.b = None

    def __str__(self):
        return 'Variable(name:{}, dtype: {})'.format(self.name, self.dtype)

    def forward_pass(self):
        return self.value
