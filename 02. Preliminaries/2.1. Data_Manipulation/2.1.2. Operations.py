############### 2.1.2. Operations ###############

# import
from mxnet import np, npx
npx.set_np()

# 기본 계산
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y    # 더하기
x - y    # 빼기
x * y    # 곱하기
x / y    # 나누기
x ** y   # 제곱

# Many more operations can be applied elementwise, including unary operators like exponentiation.
np.exp(x)

# concatenate multiple ndarrays together
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0)
np.concatenate([x, y], axis=1)

# logical statements 로 나타내기
x == y

# 모든 요소 합하기
x.sum()
np.sum(x)
