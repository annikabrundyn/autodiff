# autodiff 

[WIP]
this is a basic automatic differentiation library in python and numpy.

todos:
1. pretty code: adding tests, adding comments (@annika) - put the backward into a function
2. profiling - use the one from  (@fra)
3. cupy  (@yang)
4. baseline example <numpy> (keep as is but time it) (@annika)
5. report /slides (@travis)

maybe things we can do:
-parallelize batches
-make it an actual python package - with documentation (maybe)




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