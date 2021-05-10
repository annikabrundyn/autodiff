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


class CategoricalCrossEntropy(Layer):
    """WIP - similar to pytorch - applies softmax then ce
    """
    def __init__(self, ):
        super().__init__('CE Loss', 1)

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return self.forward(pred, target)

    def forward(self, pred: np.ndarray, target: np.ndarray, reduction='mean') -> np.ndarray:
        #self.pred = pred.clip(min=1e-8, max=None)
        bs = pred.shape[0]

        # apply stable softmax
        probs = np.exp(pred - np.max(pred))
        self.pred = probs / np.sum(probs, axis=1)[:, np.newaxis]
        #self.pred = pred.clip(min=1e-8, max=None)
        self.target = target

        # calculate loss
        loss = -np.sum(np.log(self.pred[np.arange(bs), self.target])) / bs
        return loss
        #return (np.where(self.target == 1, -np.log(self.pred), 0)).sum(axis=1)

        #loss = - np.log(self.pred[np.arange(len(self.target)), self.target])

        # if reduction == 'mean':
        #     batch_size = pred.shape[0]
        #     loss = - np.sum(loss) / batch_size
        # return loss

    def backward(self) -> np.ndarray:
        # # TODO: CHECK THIS
        # batch_size = self.pred.shape[0]
        #
        # probs = self.pred.copy()
        # probs[np.arange(len(probs)), self.target] -= 1
        #
        # return (probs/batch_size)
        bs = self.pred.shape[0]
        probs = np.copy(self.pred)
        probs[np.arange(bs), self.target] -= 1
        grad = probs/bs
        #grad = 1 / len(self.target) * probs.dot(x.T) + reg * self.W
        #np.where(self.target == 1, -1/probs, 0)
        return probs/bs


