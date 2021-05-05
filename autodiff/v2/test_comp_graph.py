import numpy as np

from node import Node
from operations import *
from utils import topological_sort
from grad import evaluate_dag, backward_diff_dag

def mlp(input=np.random.rand(10),output=np.random.rand(1),hidden_dims=[5,1]):
    
    print('input:',input.shape,'output:',output.shape)
    # variable nodes
    x = Node(value=input, name="x")
    y = Node(value=output, name="y")

    weights = list()
    for idx,h in enumerate(hidden_dims):
        if idx == 0:
            # variable nodes
            exec(f'w{idx} = Node(value=np.random.rand(h, input.shape[0]), name="w0")')
        else:
            exec(f'w{idx} = Node(value=np.random.rand(h, hidden_dims[idx-1]), name="w{idx}")')
        exec(f'weights.append(w{idx})')
        
        if idx == len(hidden_dims):
            exec(f'w{idx} = Node(value=np.random.rand(h, output.shape[0], name="w{idx}")')

    return weights

weights = mlp(input=np.random.rand(4),output=np.random.rand(1),hidden_dims=[2,2])
print('weights:')
for w in weights:
    print(w.name,w.value)
