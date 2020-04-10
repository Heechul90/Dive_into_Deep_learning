############### 2.1.5. Saving Memory ###############

# import
from mxnet import np, npx
npx.set_np()

# memoory id 저장하기
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

before = id(y)
y = y + x
id(y) == before

# we can assign the result of an operation to a previously allocated array with slice notation
z = np.zeros_like(y)   # y랑 shape은 같게 값은 0으로
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))

# if the value of x is not reused in subsequent computations, we can also use x[:] = x + y or x += y
# to reduce the memory overhead of the operation
before = id(x)
x += y
id(x) == before

