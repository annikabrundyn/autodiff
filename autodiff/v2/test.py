import numpy as np

from node import Node
from operations import *
from utils import topological_sort
from grad import evaluate_dag, backward_diff_dag

x = [0.5, 1.3]
x1 = Node(value=np.array([x[0]]), name="x1")
x2 = Node(value=np.array([x[1]]), name="x2")
x3 = Node(func=exp, parents=[x1], name="x3")
x4 = Node(func=mul, parents=[x2, x3], name="x4")
x5 = Node(func=add, parents=[x1, x4], name="x5")
x6 = Node(func=sqrt, parents=[x5], name="x6")
x7 = Node(func=mul, parents=[x4, x6], name="x7")

sorted_nodes = topological_sort(x7)
node_names = [node.name for node in sorted_nodes]

value = evaluate_dag(sorted_nodes)

backward_diff_dag(sorted_nodes)


# yhat = xw + b
# loss = (yhat - y)^2

# y = (2, 1)
# x = (2, 5)
# w = (5, 1)
# b = (2, 1)

y = Node(value=np.random.rand(2, 1), name="y")

x = Node(value=np.random.rand(2, 5), name="x")
w = Node(value=np.random.rand(5), name="w")
b = Node(value=np.random.rand(2, 1), name="b")

wx = Node(func=dot, parents=[w, x], name="wx")

yhat = Node(func=add, parents=[b, wx], name="yhat")

loss = Node(func=squared_loss, parents=[y, yhat], name="loss")

sorted_nodes = topological_sort(loss)

value = evaluate_dag(sorted_nodes)

backward_diff_dag(sorted_nodes)

print("hey")