

class Node(object):

    def __init__(self, children, name="Node"):
        self.children = children
        self.name = name

        self.id = Node.id
        Node.id += 1

    def forward(self):
        pass

    def backward(self):
        pass


    def __add__(self, other):
        from .operations import Add
        return Add(self, other)

    def __sub__(self, other):
        from .operations import Subtract
        return Subtract(self, other)

    def __mul__(self, other):
        from .operations import Multiply
        return Multiply(self, other)

    def __pow__(self, power, modulo=None):
        from .operations import Power
        return Power(self, power)

    def __floordiv__(self, other):
        from .operations import Divide
        return Divide(self, other)

    def __truediv__(self, other):
        from .operations import Divide
        return Divide(self, other)


class Variable(Node):
    def __init__(self, name):
        self.name = name
        self.value = None
        self.a = None
        self.b = None

    def __str__(self):
        return 'Variable(name:{})'.format(self.name)

    def forward_pass(self):
        return self.value
