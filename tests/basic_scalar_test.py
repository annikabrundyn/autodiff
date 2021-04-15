import autodiff as ad

# first define our variables symbolically
x1 = ad.Variable(name='x1')
x2 = ad.Variable(name='x2')

# then define symbolic expression for y
y = x1 * x2 + x1

grad_x = ad.grad(y, [x1])
print('dz/dx:', grad_x(x=2.0, y=3.0))

# calculate forward only
f_z = ad.compile(z)
print(f_z(x=2.0, y=3.0))


