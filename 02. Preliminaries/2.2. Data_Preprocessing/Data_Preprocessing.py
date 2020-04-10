############### 2.2.1. Reading the Dataset ###############

# import
import os

# saved in the d2l package for later use
def mkdir_if_not_exist(path):
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)

# we write the dataset row by row into a csv file.
data_file = 'data/house_tiny.csv'
mkdir_if_not_exist('data')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')        # Each row is a data point
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# load the raw dataset from the created csv file

# if pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)



############### 2.2.2. Handling Missing Data ###############
# NaN값은 missing values다. 이 값을 처리하기 위해서는 값을 채워 넣거나 삭제할 수 있다.
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

# Alley 컬럼의 경우 Alley_Pave와 Alley_NaN를 1과 0으로 나눔
inputs = pd.get_dummies(inputs, dummy_na=True)



############### 2.2.3. Conversion to the ndarray Format ###############
# inputs과 outputs이 numerical이라면 ndarray format이 가능하다
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X
y



############### 2.2.4. Summary ###############
# like many other extension packages in the vast ecosystem of Python, pandas can work together with ndarray
# imputation and deletion can be used to handle missing data