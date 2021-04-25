
def evaluate_dag(sorted_nodes):
  for node in sorted_nodes:
    if node.value is None:
      values = [p.value for p in node.parents]
      node.value = node.func(*values)
  return sorted_nodes[-1].value


def backward_diff_dag(sorted_nodes):
  value = evaluate_dag(sorted_nodes)
  m = value.shape[0]  # Output size

  # Initialize recursion.
  sorted_nodes[-1].grad = np.eye(m)

  for node_k in reversed(sorted_nodes):
    if not node_k.parents:
      # We reached a node without parents.
      continue

    # Values of the parent nodes.
    values = [p.value for p in node_k.parents]

    # Iterate over outputs.
    for i in range(m):
      # A list of size len(values) containing the vjps.
      vjps = node_k.func.make_vjp(*values)(node_k.grad[i])

      for node_j, vjp in zip(node_k.parents, vjps):
        node_j.grad += vjp

  return sorted_nodes