import numpy as np

def one_hot_encoder(y):
    y_ohe = y_pred[np.arange(len(y)), y]
    y_ohe[np.arange(y.size), y] = 1
    return y_ohe