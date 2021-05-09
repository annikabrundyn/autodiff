# autodiff 

[WIP]
this is a basic automatic differentiation library in python and numpy.

right now, this only works with scalars and is using forward mode AD.

TODOS:

- [ ] add Cross Entropy loss
- [ ] parallelization / gpus
- [ ] figure out why doesnt work for regression - have to have sigmoid at end/doesnt work with linear



cupy for extending to gpus:
https://cupy.dev/#features



## How to run

first install the project:
```bash
cd autodiff
pip install -e .
```

then train the model:
```bash
python train_model.py
```