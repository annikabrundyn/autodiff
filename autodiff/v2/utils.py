def dfs(node, visited):
  visited.add(node)
  for parent in node.parents:
    if not parent in visited:
      # Yield parent nodes first.
      yield from dfs(parent, visited)
  # And current node later.
  yield node


def topological_sort(end_node):
  """
  Topological sorting.
  Args:
    end_node: in.
  Returns:
    sorted_nodes
  """
  visited = set()
  sorted_nodes = []

  # All non-visited nodes reachable from end_node.
  for node in dfs(end_node, visited):
    sorted_nodes.append(node)

  return sorted_nodes