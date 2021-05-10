import numpy as np
from layer import Layer


class BCE(Layer):
    """Binary Cross-Entropy.
    We assume the loss layer is the last of the network; hence, it does not need the
    error of the following layer.
    """

    def __init__(self, ):
        super().__init__('BCE Loss', 1)

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return self.forward(pred, target)

    def forward(self, pred: np.ndarray, target: np.ndarray, reduction='mean') -> np.ndarray:
        self.pred = pred
        self.target = target

        loss = -self.target * np.log(self.pred) - (1 - self.target) * np.log(1 - self.pred)
        if reduction == 'mean':
            batch_size = pred.shape[0]
            loss = np.sum(loss) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        batch_size = self.pred.shape[0]
        val = -(self.target / self.pred) + ((1 - self.target) / (1 - self.pred))
        return (val/batch_size)


class MSE(Layer):
    """Mean square error.
    We assume the loss layer is the last of the network; hence, it does not need the
    error of the following layer.
    """

    def __init__(self):
        super().__init__('MSE Loss', 1)

    def __call__(self,  pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return self.forward(pred, target)

    def forward(self, pred: np.ndarray, target: np.ndarray, reduction='mean') -> np.ndarray:
        self.pred = pred
        self.target = target
        loss = np.power(self.pred - self.target, 2)

        if reduction == 'mean':
            batch_size = pred.shape[0]
            loss = np.sum(loss) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        batch_size = self.pred.shape[0]
        val = 2 * (self.pred - self.target)
        return (val/batch_size)


# WIP
# class CrossEntropyLoss():
#
#     def __init__(self):
#         pass
#
#     def get(self, y_pred, y):
#         """
#             Return the negative log likelihood and the error at the last layer.
#
#             Parameters:
#             - y_pred: model predictions.
#             - y: ground truth labels.
#         """
#         batch_size = y_pred.shape[1]
#         deltaL = y_pred - y
#         loss = -np.sum(y * np.log(y_pred)) / batch_size
#         return loss, deltaL



