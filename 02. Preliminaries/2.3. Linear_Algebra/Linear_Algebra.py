############### 2.3.1. Scalars ###############

# import
from mxnet import np, npx
npx.set_np()

# mxnet에서 scalar값은 ndarray로 나타난다
x = np.array(3.0)
y = np.array(2.0)
x + y, x * y, x / y, x ** y,


############### 2.3.2. Vectors ###############
# mxnet에서, we workk with vectors via 1-dimensional ndarrays.
# in general ndarrays can have arbitrary lengths, subject th the memory limits of your machine.
x = np.arange(4)
x

x[3]



############### 2.3.2.1. Length, Dimensionality and Shape ###############
# ndarray는 length를 가진다.
len(x)

# ndarray가 vector로 나타낼때, .shape을 통해서 length를 볼 수 있다.
x.shape



############### 2.3.3. Matrices ###############
A = np.arange(20).reshape(5, 4)
A

# we access a matrix's transpose via the T attribute.
A.T

# as a special type of the square matrix, a symmetric matrix A is equal to its transpose: A = A transpose.
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
B == B.T



############### 2.3.4. Tensors ###############
# images arrive as ndarrays with 3 axes corresponding to the height, width and a channel axis
# for stacking the color channels (red, green and blue)
X = np.arange(24).reshape(2, 3, 4)
X



############### 2.3.5. Basic Properties of Tensor Arithmetic ###############
A = np.arange(20).reshape(5, 4)
B = A.copy()     # Assign a copy of A to B by allocating new memory
A
A + B
A * B

# multiplying of adding a tensor by a scalar also does not change the shape of the tensor,
# where each element of the operand tensor will be added or multiplied by the scalar.
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X
(a * X).shape



############### 2.3.6. Reduction ###############
x = np.arange(4)
x, x.sum()

# we can express sums over the elements of tensors of arbitrary shape
A.shape
A.sum()

# we can specify the axes along which the tensor is reduced via summation.
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0
A_sum_axis0.shape

A_sum_axis1 = A.sum(axis=1)
A_sum_axis1
A_sum_axis1.shape

# reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix.
A.sum(axis=[0, 1])     # Same as A.sum()
A.sum()

# we can just call mean on tensors of arbitrary shape.
A.mean(), A.sum() / A.size

# like sum, mean can also reduce a tensor along the specified axes.
A.mean(axis=0), A.sum(axis=0) / A.shape[0]



############### 2.3.6.1. Non-Reduction Sum ###############
# sometimes is can be useful to keep the number of axes unchanged when invoking sum or mean by setting keepdims=True.
sum_A = A.sum(axis=1, keepdims=True)

# for instance, siince sum_A still keeps its 2 axes after summing each row, we can divide A by sum_A with broadcasing.
A / sum_A

# we can call the cumsum function
# this function will not reduce the input tensor along any axis.
A.cumsum(axis=0)



############### 2.3.7. Dor Products ###############
y = np.ones(4)
x
y
np.dot(x, y)

# we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:
np.sum(x * y)



############### 2.3.8. Matrix-Vector Products ###############
# we can begin to understand matrix-vector products
A.shape, x.shape, np.dot(A, x)



############### 2.3.9. Matrix-Matrix Multiplication ###############
# if you have gotten the hang of dot products and matrix-vector products, then matrix-matrix multiplication should be straightforward.
B = np.ones(shape=(4, 3))
np.dot(A, B)



############### 2.3.10. Norms ###############
u = np.array([3, -4])
np.linalg.norm(u)
np.abs(u).sum()
np.linalg.norm(np.ones((4, 9)))



############### 2.3.10.1. Norms and Objectives ###############
# while we do not want to get too far ahead of ourselves, we can plant some intuition already about why these concepts are useful
# in deep learning, we are often trying to solve optimization provlems



############### 2.3.11. More on Linear Algebra ###############
# in just this section, we have taught you all the linear algebra that you will need to understand a remarkable chunk of modern deep learning



############### 2.3.12. Summary ###############
# Scalars, vectors, matrices and tensors are basic mathematical objects in linear algebra.
# Vectors generalize scalrs and matrices generalize vectors
# In the ndarray representation, scalar, vectors, matrices and tensors have 0,1,2 and an arbitrary number of axes, respectively
# A tensor can be reduced along the specified axes by sum and mean
# Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication.
# In deep learning, we often work with norms such as the L1 norm, the L2 norm and the Frobenius norm
# we can perform a variety of operations over scalars, vectors, matrices and tensors with ndarray functions