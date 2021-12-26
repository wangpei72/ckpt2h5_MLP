import numpy as np

from adf_data.bank import bank_data
from adf_data.census import census_data
from cov.utils_of_all import *

np.set_printoptions(threshold=np.inf)

X, Y, input_shape, nb_classes = census_data()

x_train = X[0:80]
x_test = np.load("local_samples.npy")
b = np.ones((870,1))
x_test=np.c_[x_test,b]
print(x_test.shape)
print(x_train.shape)



