import numpy as np

min_max = np.load("result/census_8_8000/global_samples.npy")
np.set_printoptions(threshold=np.inf)
print(min_max)
print(min_max.size)