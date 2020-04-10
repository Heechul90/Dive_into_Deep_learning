############### 2.1.4. Indexing and Slicing ###############

# import
from mxnet import np, npx
npx.set_np()

# 인텍싱 및 슬라이싱
x = np.arange(12).reshape(3, 4)
x
x[-1]
x[1:3]

# metrix의 elements 수정하기
x[1, 2] = 9
x

# multiple elements를 똑같은 값으로 바꾸고 싶으면 인덱싱으로 수정
x[0:2, :] = 12
x




