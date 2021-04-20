import functools
import numpy as np
import collections
from utils import reverse_topo_sort
from ops import Add
from node import Variable


def grad(top_node, wrt_list, previous_grad=None):
    
    assert isinstance(wrt_list, list) or isinstance(wrt_list, tuple)
    if previous_grad is None:
        previous_grad = Variable(np.ones(top_node.shape), name=add_sum_name(top_node))

    # if call dct with nonexistent key, the key value pair (key, Variable(0)) will be added
    dct = collections.defaultdict(lambda: Variable(0))
    dct[top_node] += previous_grad  # add the incoming gradient for the top node

    def add_partials(dct, node):
        for child in set(node.children):  # calc. all partial derivs w.r.t. each child and add them to child's grads
            dct[child] += node.partial_derivative(wrt=child, previous_grad=dct[node])
        return dct

    rev_graph = reverse_topo_sort(top_node)
    for node in rev_graph:
        add_partials(dct, node)

    return [dct[wrt] for wrt in wrt_list]


def add_sum_name(node):
    return "'" + node.name + "' grad_sum"
