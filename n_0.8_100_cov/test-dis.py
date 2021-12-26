import numpy as np

from adf_data.bank import bank_data
from adf_data.census import census_data
from adf_data.credit import credit_data
from cov.utils_of_all import *

# np.set_printoptions(threshold=np.inf)

model_name = "0.8-census"

model = tf.keras.models.load_model("0.8-census.h5")
X, Y, input_shape, nb_classes = census_data()
p=X.shape[0]*0.8
p=int(p)
x_train = X[:p]
npy = ".census/9/local_samples.npy"
x_test = np.load(npy)

b = np.random.randint(0,1, size=(1, 1000))
x_test = np.insert(x_test[:1000], 8, b, axis=1)

getMin_Max(model, x_train, model_name)
coverDic = {}
cover_path = model_name + "-dis-1000.csv"
min_max = model_name + ".npy"
print(x_test.shape)

for i in range(10):
    data = x_test.copy()
    # data = tf.random.shuffle(data)
    # data = data[:20000]
    output = getOutPut(model,  data)
    print(output.shape)
    nc, ac = neuronCover(output)
    coverDic["神经元覆盖率"] = nc
    knc = KMNCov(output, 100, min_max)
    coverDic["K-多节神经元覆盖率"] = knc
    nbc, Upper = NBCov(output, min_max)
    coverDic["神经元边界覆盖率"] = nbc
    snc = SNACov(output, Upper)
    coverDic["强神经元激活覆盖率"] = snc
    tknc = TKNCov(model, data, 2)
    coverDic["top-k神经元覆盖率"] = tknc
    if not os.path.exists(cover_path):
        create_csv(cover_path, coverDic)
    else:
        append_csv(cover_path, coverDic)


