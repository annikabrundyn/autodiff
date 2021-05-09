from model import Model
from layer import *
from loss import BCE, MSE
from sklearn.datasets import make_classification, load_iris, load_boston, load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


THRESHOLD = 0.5

### Make dataset
X, Y = make_classification(n_samples=10000, n_features=3, n_informative=2, n_redundant=1, random_state=42)


#scaler = preprocessing.MinMaxScaler().fit(X_train)
#X = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


### Define the model
model = Model()
model.add(Linear(3, 10))
model.add(ReLU(10))
model.add(Linear(10, 5))
model.add(ReLU(5))
model.add(Linear(5, 1))
model.add(Sigmoid(1))



losses = []

for epoch in range(1000000):

    # forward
    # our model takes in the shape (feats, samples)
    pred = model(X.T)

    loss_f = BCE(pred, Y)

    error = loss_f()
    losses.append(error)

    gradient = loss_f.backward()

    # Backpropagation
    for i, _ in reversed(list(enumerate(model.layers))):
        if model.layers[i].type != 'Linear':
            gradient = model.layers[i].backward(gradient)
        else:
            gradient, dW, dB = model.layers[i].backward(gradient)
            model.layers[i].optimize(dW, dB, 0.05)

    if epoch % 10000:
        print("current loss: ", error)
        pred_label = (pred >= THRESHOLD).astype('int')
        print("current accuracy: ", accuracy_score(Y, pred_label.squeeze()))

