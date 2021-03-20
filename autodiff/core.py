import numpy as np


class Tensor:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name


class Node:
    def __init__(self, fn):
        self.fn = fn