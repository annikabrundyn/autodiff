class Node(object):

  def __init__(self, value=None, func=None, parents=None, name=""):
    # Value stored in the node.
    self.value = value
    # Function producing the node.
    self.func = func
    # Inputs to the function.
    self.parents = [] if parents is None else parents
    # Unique name of the node (for debugging and hashing).
    self.name = name
    # Gradient / Jacobian.
    self.grad = 0
    if not name:
      raise ValueError("Each node must have a unique name.")

  def __hash__(self):
    return hash(self.name)

  def __repr__(self):
    return "Node(%s)" % self.name
