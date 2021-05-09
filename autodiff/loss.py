import numpy as np
from layer import Layer


class MSE(Layer):
    """Mean square error.
    We assume the loss layer is the last of the network; hence, it does not need the
    error of the following layer.
    """

    def __init__(self, pred: np.ndarray, target: np.ndarray):
        super().__init__('MSE Loss', 1)
        self.pred = pred
        self.target = target

    def __call__(self) -> np.ndarray:
        return self.forward(self.pred, self.target)

    def forward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.power(self.pred - self.target, 2).mean()

    def backward(self) -> np.ndarray:
        return 2 * (self.pred - self.target).mean()


class BCE(Layer):
    """Binary Cross-Entropy.
    We assume the loss layer is the last of the network; hence, it does not need the
    error of the following layer.
    """

    def __init__(self, pred: np.ndarray, target: np.ndarray):
        super().__init__('BCE Loss', 1)
        self.pred = pred
        self.target = target

    def __call__(self) -> np.ndarray:
        return self.forward()

    def forward(self) -> np.ndarray:
        n = len(self.target)
        loss = np.nansum(-self.target * np.log(self.pred) - (1 - self.target) * np.log(1 - self.pred)) / n

        return np.squeeze(loss)

    def backward(self) -> np.ndarray:
        n = len(self.target)
        return (-(self.target / self.pred) + ((1 - self.target) / (1 - self.pred))) / n