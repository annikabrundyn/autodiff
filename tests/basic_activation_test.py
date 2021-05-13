import numpy as np
import autodiff as ad


# test activation layers
def test_relu(data: np.ndarray):
    relu_layer = ad.ReLU()
    assert np.array_equal(relu_layer.forward(data), np.maximum(0, data))


def test_tanh(data: np.ndarray):
    tanh_layer = ad.TanH()
    assert np.array_equal(tanh_layer.forward(data), np.tanh(data))


def test_sigmoid(data: np.ndarray):
    sigmoid_layer = ad.Sigmoid()
    assert np.array_equal(sigmoid_layer.forward(data), 1 / (1 + np.exp(-data)))


test_array = np.array([1, 3, -2])
test_relu(test_array)
test_tanh(test_array)
test_sigmoid(test_array)
