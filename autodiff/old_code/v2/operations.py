import numpy as np

# Basic functions and their VJPs.

def dot(x, W):
  return np.dot(W, x)


def dot_make_vjp(x, W):
  def vjp(u):
    return W.T.dot(u), np.outer(u, x)
  return vjp


dot.make_vjp = dot_make_vjp


def relu(x):
  return np.maximum(x, 0)


def relu_make_vjp(x):
  gprime = np.zeros(len(x))
  gprime[x >= 0] = 1

  def vjp(u):
    return u * gprime,  # The comma is important.

  return vjp


relu.make_vjp = relu_make_vjp


def squared_loss(y_pred, y):
  # The code requires every output to be an array.
  return np.array([0.5 * np.sum((y - y_pred) ** 2)])


def squared_loss_make_vjp(y_pred, y):
  diff = y_pred - y

  def vjp(u):
    return diff * u, -diff * u

  return vjp


squared_loss.make_vjp = squared_loss_make_vjp


def add(a, b):
  return a + b


def add_make_vjp(a, b):
  gprime = np.ones(len(a))

  def vjp(u):
    return u * gprime, u * gprime

  return vjp


add.make_vjp = add_make_vjp


def mul(a, b):
  return a * b


def mul_make_vjp(a, b):
  gprime_a = b
  gprime_b = a

  def vjp(u):
    return u * gprime_a, u * gprime_b

  return vjp


mul.make_vjp = mul_make_vjp


def exp(x):
  return np.exp(x)


def exp_make_vjp(x):
  gprime = exp(x)

  def vjp(u):
    return u * gprime,

  return vjp


exp.make_vjp = exp_make_vjp


def sqrt(x):
  return np.sqrt(x)


def sqrt_make_vjp(x):
  gprime = 1. / (2 * sqrt(x))

  def vjp(u):
    return u * gprime,

  return vjp


sqrt.make_vjp = sqrt_make_vjp
