import autodiff as ad
import numpy as np

x = np.random.rand(2, 1, 28, 28)
y = np.random.rand(2, 1)


lr = 0.01

model = ad.Model()
model.add(ad.Conv2D(in_channels=1, out_channels=3, filter_size=3))
model.add(ad.ReLU())
model.add(ad.Conv2D(in_channels=3, out_channels=1, filter_size=3))
model.add(ad.Flatten())  #(2, 1, 4, 4)
model.add(ad.Linear(576, 50))
model.add(ad.ReLU())
model.add(ad.Linear(50, 1))
model.add(ad.Sigmoid(1))

criterion = ad.BCE()

# forward
y_pred = model.forward(x)

loss = criterion.forward(y_pred, y)

# backprop
model.backward(criterion)

# update weights
model.update_params_sgd(lr=0.01)

print("hi")