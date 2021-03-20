import numpy as np


class Node:
    def __init__(self, children, name="Node"):
        self.children = [child if isinstance(child, Node) else Variable(child) for child in children]
        self.name = name

    def __add__(self, other):
        return Add(self, other)



class Variable(Node):
    def __init__(self, value, name=None):
        self.value = value
        self.name = name
        self._value = value

    def __str__(self):
        return f"My variable with name: {self.name} and value: {self._value}"

    def _eval(self):
        return self._value

    def _partial_derivative(self, wrt, previous_grad):
        if self == wrt:
            return previous_grad
        return 0




class Add(Node):
    def __init__(self, *elems, name="Add"):
        super().__init__(list(elems), name)

    def _eval(self):
        return np.array(sum([elem() for elem in self.children]))

    def _partial_derivative(self, wrt, previous_grad):
        wrt_count = self.children.count(wrt)
        grad = previous_grad * Variable(wrt_count)
        return grad


print("hey")