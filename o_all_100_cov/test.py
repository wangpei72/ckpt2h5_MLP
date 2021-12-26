from adf_data.bank import bank_data
from adf_data.census import census_data
from adf_data.credit import credit_data
from cov.utils_of_all import *

# np.set_printoptions(threshold=np.inf)



model_name = "all-credit"

model = tf.keras.models.load_model("all-credit.h5")
X, Y, input_shape, nb_classes = credit_data()
p=X.shape[0]*0.8
p=int(p)
x_train = X[:p]
x_test = X[p:]

# x_test = np.load("local_samples.npy")
# b = np.ones((870,1))
# x_test=np.c_[x_test,b]

getMin_Max(model, x_train, model_name)
coverDic = {}
cover_path = model_name + "-normal.csv"
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


