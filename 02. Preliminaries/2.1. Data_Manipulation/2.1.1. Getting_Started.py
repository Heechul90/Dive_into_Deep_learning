############### 2.1.1. Getting Started ###############

# import
from mxnet import np, npx
npx.set_np()

# vector 생성
x = np.arange(12)
x
x.shape     # shape 확인
x.size      # size 확인

# shape chahe
x = x.reshape(3, 4)
x
x.shape

# 자동으로 남은 값들의 dimention 작동
x.reshape(-1, 4)
x.reshape(3, -1)

# empty 메모리의 덩어리를 가져오는 mothod
np.empty((3, 4))

# 0으로 값을 채우기
np.zeros((2, 3, 4))

# 1로 값을 채우기
np.ones((2, 3, 4))

# 평균이 0이고, 표준편차가 1인 (3, 4) dimention 만들기
np.random.normal(0, 1, size=(3, 4))

# 직접 리스트로 dimention 만들기
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
