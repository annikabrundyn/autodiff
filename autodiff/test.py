from node import Node, Variable
from ops import *
from grad import grad

a = Variable(3, name="a")
b = Variable(4, name="b")
c = Variable(5, name="c")

z = (a*b*c)

a_grad = grad(z, [a])

print('hey')