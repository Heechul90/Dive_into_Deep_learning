############### 2.1.3. Broadcasting Mechanism ###############

# import
from mxnet import np, npx
npx.set_np()

# we broadcast along an axis where an array initially only has length 1
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a
b

# a와 b를 더할려고 하면 두개의 shape이 맞지 않다.
# 더 할경우 row와 column의 가장 긴 길이로 결과물이 나오는데, 여기서는 (3, 2)
# a는 컬럼을 복제해서 (3, 2)를 만들고 계산한다.
a + b

