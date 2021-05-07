import numpy as np
from layer import Layer


class MSE(Layer):
    """Mean square error.
    We assume the loss layer is the last of the network; hence, it does not need the
    error of the following layer.
    """

    def __init__(self):
        self.type = 'MSE'

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.power(self.pred - self.target, 2).mean()

    def backward(self):
        return 2 * (self.pred - self.target).mean()

    def __call__(self, pred, target):
        return self.forward(pred, target)



class BCE(Layer):
    """Binary Cross-Entropy.
    We assume the loss layer is the last of the network; hence, it does not need the
    error of the following layer.
    """

    def __init__(self):
        self.type = 'BCE'

    def forward(self, pred, target):
        self.pred = pred
        self.target = target

        n = len(self.target)
        loss = np.nansum(-self.target * np.log(self.pred) - (1 - self.target) * np.log(1 - self.pred)) / n

        return np.squeeze(loss)

    def backward(self):
        n = len(self.target)
        return (-(self.target / self.pred) + ((1 - self.target) / (1 - self.pred))) / n

    def __call__(self, pred, target):
        return self.forward(pred, target)


