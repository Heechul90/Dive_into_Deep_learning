############### 2.1.6. Conversion to Other Python Objects ###############

# import
from mxnet import np, npx
npx.set_np()

# Converting an MXNet ndarray to a NumPy ndarray, or vice versa
x = np.arange(12).reshape(3, 4)
a = x.asnumpy()
b = np.array(a)
type(a), type(b)

# to convert a size-1 ndarray to a Python scalar, we can invoke the item function or Python's built-in functions.
a = np.array([3.5])
a, a.item(), float(a), int(a)