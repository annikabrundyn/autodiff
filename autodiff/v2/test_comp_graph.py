import numpy as np

from node import Node
from operations import *
from utils import topological_sort
from grad import evaluate_dag, backward_diff_dag

np.random.seed(0)

def mlp(input,output,hidden_dims):
    
    print('input:',input.shape,'output:',output.shape)
    # variable nodes
    x = Node(value=input, name="x")
    y = Node(value=output, name="y")

    weights = list()
    for idx,h in enumerate(hidden_dims):
        
        if idx == 0:
            # variable nodes
            # exec(f'w{idx} = Node(value=np.random.rand(h, input.shape[0]), name="w0")')
            exec(f'w{idx} = Node(value=np.full((h, input.shape[0]),idx+1), name="w0")')
        else:
            # exec(f'w{idx} = Node(value=np.random.rand(hidden_dims[idx-1],h), name="w{idx}")')
            exec(f'w{idx} = Node(value=np.full((hidden_dims[idx-1],h),idx+1), name="w{idx}")')
        
        exec(f'weights.append(w{idx})')

        if idx+1 == len(hidden_dims):
            # exec(f'w{idx} = Node(value=np.random.rand(hidden_dims[idx-1],h), name="w{idx}")')
            exec(f'w{idx} = Node(value=np.full((hidden_dims[idx-1],h),idx+1), name="w{idx}")')
            exec(f'weights.append(w{idx})')
            # exec(f'w{idx+1} = Node(value=np.random.rand(h,output.shape[0]), name="w{idx+1}")')
            exec(f'w{idx+1} = Node(value=np.full((h,output.shape[0]),idx+2), name="w{idx+1}")')
            exec(f'weights.append(w{idx+1})')
        
    layers=list()
    for idx,_ in enumerate(hidden_dims):
        if idx == 0:
            # operation node
            exec(f'l0 = Node(func=dot, parents=[x, w0], name="l0")')
        else:
            exec(f'l{idx} = Node(func=dot, parents=[w{idx},l{idx-1}], name="l{idx}")')
        
        exec(f'layers.append(l{idx})')

    if idx+1 == len(hidden_dims):
        exec(f'y_pred = Node(func=dot, parents=[w{idx+1},l{idx}], name="y_pred")')
        exec(f'layers.append(y_pred)')

    outer_l = (layers[-1])
    L = Node(func=squared_loss, parents=[outer_l, y], name="L")

    sorted_nodes = topological_sort(L)
    # print('sorted nodes:',sorted_nodes)

    value = evaluate_dag(sorted_nodes)
    print('Forward pass evaluates to:',value)
    # backward_diff_dag(sorted_nodes)

    return weights, layers#, backward_diff_dag

x_numpy = np.full(4,1)
y_numpy = np.full(2,100)

weights,layers = mlp(input=x_numpy,output=y_numpy,hidden_dims=[3,3,2])
# print('weights:')
# for w in weights:
#     print(w.name,w.value)

# print('layers:')
# for l in layers:
#     print(l.name,l.parents)

# example with fixed weights
w0 = np.full((3,4),1)
w1 = np.full((3,3),2)
w2 = np.full((2,3),3)
w3 = np.full((2,2),4)

l0 = np.dot(w0,x_numpy)
print('l0.shape',l0.shape)
l1 = np.dot(w1,l0)
print('l1.shape',l1.shape)
l2 = np.dot(w2,l1)
print('l2.shape',l2.shape)
y_pred = np.dot(w3,l2)
print('y_pred',y_pred)

print('Numpy forward pass evaluates to:', 0.5*np.sum((y_pred-y_numpy)**2) )

