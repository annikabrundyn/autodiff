import numpy as np
from autodiff.layer import Layer


class BinaryCrossEntropy(Layer):
    """
    Binary Cross-Entropy.
    """
    def __init__(self, ):
        super().__init__('BCE Loss')

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


class MeanSquareError(Layer):
    """
    Mean square error.
    """
    def __init__(self):
        super().__init__('MSE Loss')

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
    """
    CategoricalCrossEntropy - Applies softmax then NLL loss (similar to PyTorch)
    """
    def __init__(self, ):
        super().__init__('CE Loss')

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return self.forward(pred, target)

    def forward(self, pred: np.ndarray, target: np.ndarray, reduction='mean') -> np.ndarray:
        # apply stable softmax
        probs = np.exp(pred - np.max(pred))
        self.pred = probs / np.sum(probs, axis=1)[:, np.newaxis]
        self.target = target

        if reduction == 'mean':
            bs = pred.shape[0]
            loss = -np.sum(np.log(self.pred[np.arange(bs), self.target])) / bs

        return loss

    def backward(self) -> np.ndarray:
        bs = self.pred.shape[0]
        probs = np.copy(self.pred)
        probs[np.arange(bs), self.target] -= 1
        return probs/bs


