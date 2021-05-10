import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

THRESHOLD = 0.5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(3, 10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 5)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.linear1(input)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        return out


X, Y = make_classification(n_samples=10000, n_features=3, n_informative=2, n_redundant=1, random_state=42)
X, Y = compare_pytorch.from_numpy(X), compare_pytorch.from_numpy(Y)
Y = Y.type(compare_pytorch.FloatTensor)
model = Net()

optimizer = compare_pytorch.optim.SGD(model.parameters(), lr=0.05)
loss_func = compare_pytorch.nn.BCELoss()
losses = []
for epoch in range(1000):
    pred = model(X.float())
    error = loss_func(pred.squeeze(), Y)
    losses.append(error)

    optimizer.zero_grad()
    error.backward()
    optimizer.step()

    if (epoch % 100) == 0:
        print("current loss: ", error)
        pred_label = (pred.detach().numpy() >= THRESHOLD).astype('int')
        print("current accuracy: ", accuracy_score(Y, pred_label.squeeze()))
