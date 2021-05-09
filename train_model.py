import autodiff as ad
#from layer import *
#from loss import BCE
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


THRESHOLD = 0.5

### Make dataset
X, Y = make_classification(n_samples=10000, n_features=3, n_informative=2, n_redundant=1, random_state=42)


#scaler = preprocessing.MinMaxScaler().fit(X_train)
#X = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


### Define the model
model = ad.Model()
model.add(ad.Linear(3, 10))
model.add(ad.ReLU(10))
model.add(ad.Linear(10, 5))
model.add(ad.ReLU(5))
model.add(ad.Linear(5, 1))
model.add(ad.Sigmoid(1))


### Train the model
losses = []

for epoch in range(1000000):

    # forward
    # our model takes in the shape (feats, samples)
    pred = model(X.T)

    loss_f = ad.BCE(pred, Y)

    error = loss_f()
    losses.append(error)

    # Backpropagation - could implement our own optimizer?
    model.update_weights(loss_f, lr=0.05)

    if epoch % 10000:
        print("current loss: ", error)
        pred_label = (pred >= THRESHOLD).astype('int')
        print("current accuracy: ", accuracy_score(Y, pred_label.squeeze()))

