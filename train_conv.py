import autodiff as ad
import numpy as np

x = np.random.rand(2, 1, 28, 28)
y = np.random.rand(2, 1)


lr = 0.01

# TODO: double check what to use for multiclass classification softmax --> cross entropy/nll??

model = ad.Model()
model.add(ad.Conv2D(in_channels=1, out_channels=3, filter_size=3))
model.add(ad.Conv2D(in_channels=3, out_channels=1, filter_size=3))
model.add(ad.Flatten())  #(2, 1, 4, 4)
model.add(ad.Linear(576, 50))
model.add(ad.Linear(50, 1))
model.add(ad.Sigmoid(1))

criterion = ad.BCE()

# forward
y_pred = model.forward(x)


loss_value = criterion.forward(y_pred, y)
deltaL = criterion.backward()


grads = {}

for i, layer in reversed(list(enumerate(model.layers))):
    if layer.type == "Softmax":
        deltaL = model.layers[i].backward(y_pred, y)
    elif layer.type == "Flatten" or layer.type == "Sigmoid":
        deltaL = model.layers[i].backward(deltaL)
    elif layer.type == "Linear" or layer.type == "Conv":
        deltaL, dW, db = model.layers[i].backward(deltaL)
        #TODO: separate backward and update into two steps with optimizer
        #model.layers[i].W['val'] = model.layers[i].W['val'] - lr * dW
        #model.layers[i].b['val'] = model.layers[i].b['val'] - lr * db



print("hi")