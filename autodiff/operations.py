from .nodes import Node


def is_valid_node(a):
    return type(a) != int and type(a) != float


class Add(Node):

    def __init__(self, a, b):
        super(Add, self).__init__()
        self.a = a
        self.b = b
        self.name = 'Add'
        self.f = None

    def __str__(self):
        return 'Add(a:{}, b:{})'.format(self.a, self.b)

    def __add__(self, other):
        return Add(self, other)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a + forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz, wrt):
        dx = 1
        return dz * dx


class Multiply(Node):

    def __init__(self, a, b):
        super(Multiply, self).__init__()
        self.a = a
        self.b = b
        self.name = 'Multiply'
        self.f = None

    def __str__(self):
        return 'Multiply(a:{}, b:{})'.format(self.a, self.b)

    def __mul__(self, other):
        return Multiply(self, other)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a * forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz, wrt):
        # df/dx = ax = a

        # backward_a = self.a.backward_pass(dz, wrt) if is_valid_node(self.a) else self.a
        # backward_a = self.a.backward_pass(dz, wrt) if is_valid_node(self.a) else self.a
        dx = self.a
        if wrt == self.a:
            dx = self.b

        # df/dx = x*x = x^2 = 2x
        if self.a == self.b:
            dx = 2 * dx.value

        if is_valid_node(dx):
            dx = dx.value

        return dz * dx


class Power(Node):

    def __init__(self, a, b):
        super(Power, self).__init__()
        self.a = a
        self.b = b
        self.name = 'Power'
        self.f = None

    def __str__(self):
        return 'Power(a:{}, b:{})'.format(self.a, self.b)

    def __pow__(self, power, modulo=None):
        return Power(self, power)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a ** forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz):
        dx = 1
        return dz * dx


class Divide(Node):

    def __init__(self, a, b):
        super(Divide, self).__init__()
        self.a = a
        self.b = b
        self.name = 'Divide'
        self.f = None

    def __str__(self):
        return 'Divide(a:{}, b:{})'.format(self.a, self.b)

    def __floordiv__(self, other):
        return Divide(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a / forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz):
        dx = 1
        return dz * dx


class Subtract(Node):

    def __init__(self, a, b):
        super(Subtract, self).__init__()
        self.a = a
        self.b = b
        self.name = 'Subtract'
        self.f = None

    def __str__(self):
        return 'Subtract(a:{}, b:{})'.format(self.a, self.b)

    def __sub__(self, other):
        return Subtract(self, other)

    def forward_pass(self):
        forward_a = self.a.forward_pass() if is_valid_node(self.a) else self.a
        forward_b = self.b.forward_pass() if is_valid_node(self.b) else self.b
        ans = forward_a - forward_b
        self.f = ans
        return ans

    def backward_pass(self, dz):
        dx = 1
        return dz * dx

