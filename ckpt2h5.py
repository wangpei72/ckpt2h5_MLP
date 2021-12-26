import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import numpy as np
def load_weights_from_ckpt(ckpt_meta_path):
    PATH_REL_META = ckpt_meta_path

    # start tensorflow session
    with tf.Session() as sess:
        # import graph
        saver = tf.train.import_meta_graph(PATH_REL_META)

        # load weights for graph
        saver.restore(sess, PATH_REL_META[:-5])

        # get all global variables (including model variables)
        vars_global = tf.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_vars = {}
        print(vars_global)
        for var in vars_global:
            # print(var)
            try:
                model_vars[var.name] = var.eval()
            except:
                print("For var={}, an exception occurred".format(var.name))
    return model_vars

# 创建模型并加载预训练的ckpt模型的权重
def create_h5_and_load_weights_from_ckpt(ckpt_meta_path):
    # model_vas的获取一定要在模型建立之前,不然会报错
    # 加载所有的权重
    model_vars = load_weights_from_ckpt(ckpt_meta_path)

    # 查看key
    print("xxx")
    print("各层权重的键为：",model_vars.keys())
    print("xxx")
    # exit(0)

    # 创建模型
    model = keras.Sequential()
    Input = (None, 13)
    layer1 = Dense(units=64, input_shape=Input, activation='relu')
    model.add(layer1)
    layer2 = Dense(units=32, activation='relu')
    model.add(layer2)

    layer3 = Dense(units=16, activation='relu')
    model.add(layer3)

    layer4 = Dense(units=8, activation='relu')
    model.add(layer4)

    layer5 = Dense(units=4, activation='relu')
    model.add(layer5)

    layer6 = Dense(units=2, activation='softmax')
    model.add(layer6)

    # 查看模型
    model.summary()


    # 加载权重
    layer1.set_weights([model_vars['linear/kernel:0'], model_vars['linear/bias:0']])
    layer2.set_weights([model_vars['linear_1/kernel:0'], model_vars['linear_1/bias:0']])
    layer3.set_weights([model_vars['linear_2/kernel:0'], model_vars['linear_2/bias:0']])
    layer4.set_weights([model_vars['linear_3/kernel:0'], model_vars['linear_3/bias:0']])
    layer5.set_weights([model_vars['linear_4/kernel:0'], model_vars['linear_4/bias:0']])
    layer6.set_weights([model_vars['linear_5/kernel:0'], model_vars['linear_5/bias:0']])

    # 测试输出
    # x = np.ones(shape=[3, 100], dtype=np.float32)
    # y = model.predict(x)
    # print(y)

    # 保存模型
    model.save("./ckpt2h2_of_test_model.h5")
    #测试结束
    print("*"*20)
    print("运行结束，模型保存在当前目录下！")
    print("*"*20)
    exit(0)

if __name__ == '__main__':
    # ckpt的模型图结构文件的位置
    ckpt_meta_path = r"./ckpt/test.model.meta"
    create_h5_and_load_weights_from_ckpt(ckpt_meta_path)