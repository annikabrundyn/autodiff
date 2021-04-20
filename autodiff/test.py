from node import Node, Variable
from ops import *
from grad import grad

x = Variable(3, name="x")
y = Variable(4, name="y")

z = x * y + Exp(x)

x_grad = grad(z, [x])

print('hey')