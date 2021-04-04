import autodiff as ad

x = ad.Variable(ad.int32, name='x')
y = ad.Variable(ad.int32, name='y')

z = x*y

grad_f = ad.grad(f)
print(grad_f(2.0, 3.0))






# from autodiff.nodes import tensor
# from autodiff.autodiff import gradients, compile
#
# x = tensor('int32', name='x')
# y = tensor('int32', name='y')
#
# z = (x*x)
#
# grads = gradients(z, [x])
# print("done")