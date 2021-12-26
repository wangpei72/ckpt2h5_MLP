import numpy as np

from adf_data.bank import bank_data
from adf_data.census import census_data
from cov.utils_of_all import *

np.set_printoptions(threshold=np.inf)

X, Y, input_shape, nb_classes = census_data()
p=X.shape[0]*0.8
p=int(p)

x_test = np.load("./census/9/suc_idx.npy")
#b = np.ones((870,1))
#x_test=np.c_[x_test,b]
print(x_test.shape)
print(x_test.shape[0])


# b = np.random.randint(0,1, size=(1, 43717))
# x_test = np.insert(x_test, 8, b, axis=1)
# print(x_test.shape)
# print(x_test.shape[0])


