import autodiff as ad

x = ad.Variable('int32', name='x')
y = ad.Variable('int32', name='y')

z = (x*x) + y

grad_x = ad.grad(z, [x])
print('dz/dx:', grad_x(x=2.0, y=3.0))

# calculate forward only
f_z = ad.compile(z)
print(f_z(x=2.0, y=3.0))