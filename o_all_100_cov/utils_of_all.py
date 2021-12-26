import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from skimage.exposure import rescale_intensity
import os
import cv2
import math
import matplotlib as mpl
import matplotlib.pyplot as plt



def getMin_Max(model,data, min_max_file_name):

    output = getOutPut(model, data)
    min_max = np.zeros((output.shape[-1], 2))
    for i in range(output.shape[-1]):
        min_max[i][0], min_max[i][1] = output[...,i].min(),output[...,i].max()

    np.save(min_max_file_name, min_max)
    print("神经元上下边界生成成功")

def getLayerOutPut(model,data):
    trainableLayers = get_trainable_layers(model)
    layers_output = []
    for l in trainableLayers:
        str1 = str(model.layers[int(l)])
        print(str1)
        if "Dense" not in str1 and "convolutional" not in str1:
            continue
        sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=l).output)
        layer_output = sub_model.predict(data)

        if layer_output.ndim > 2:
            layer_output = deal_cover_conv(layer_output)
        print(layer_output.shape)
        # Cov2D_output.append(output)
        layers_output.append(layer_output)

    return layers_output


def getOutPut(model, data):
    trainableLayers = get_trainable_layers(model)
    print("trainableLayers:", trainableLayers)
    layers_output = []
    for l in trainableLayers:
        print("l:", l)
        str1 = str(model.layers[int(l)])
        print("str1:", str1)

        sub_model = tf.keras.Model(inputs=model.input, outputs = model.get_layer(index=l).output)
        layer_output = sub_model.predict(data)
        print("layer_output1:", layer_output)

        if layer_output.ndim > 2:
            layer_output = deal_cover_conv(layer_output)
            print("layer_output2:", layer_output)
        print("layer_output.shape:",layer_output.shape)
            # Cov2D_output.append(output)
        layers_output.append(layer_output)
    print("layer_output3:", layers_output)
    output = layers_output[0]
    print("======================")
    for i in range(len(layers_output) - 1):
        i += 1
        output = np.concatenate([output, layers_output[i]], axis=1)
        print(layers_output[i].shape)
    return output

def generateErrorCsv(csv_file,nums):
    list = ["网络层数","神经元序号","可疑度"]

    with open(csv_file,"w",newline='') as cover_file:
        writer = csv.writer(cover_file)
        writer.writerow(list)
        for i in range(len(nums)):
            writer.writerow(nums[i])

def deal_cover_conv(output):
      m = output.shape[0]
      n = output.shape[-1]
      temp = np.zeros((m,n))
      for i in range(m):
            for j in range(n):
                  temp[i][j]= np.mean(output[i][...,j])
      return temp

def create_csv(csv_file, dic):

    cover_type = ["神经元覆盖率", "K-多节神经元覆盖率", "神经元边界覆盖率", "强神经元激活覆盖率", "top-k神经元覆盖率"]

    list_value = []
    for i in range(len(cover_type)):
        list_value.append(dic.get(cover_type[i],-1))
    list_tittle = []

    for i in range(len(cover_type)):
        list_tittle.append(cover_type[i])
    with open(csv_file,"w",newline='') as cover_file:
        writer = csv.writer(cover_file)
        writer.writerow(list_tittle)
        writer.writerow(list_value)

def append_csv(csv_file,dic):
    cover_type = ["神经元覆盖率", "K-多节神经元覆盖率", "神经元边界覆盖率", "强神经元激活覆盖率", "top-k神经元覆盖率"]

    list_value = []
    for i in range(len(cover_type)):
        list_value.append(dic.get(cover_type[i], -1))

    with open(csv_file, "a+", newline='') as cover_file:
        writer = csv.writer(cover_file)
        writer.writerow(list_value)


#(1) Translation params in (1,10)
def image_translation(img, params):
    params = [params * 10, params * 10]

    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

#(2)Scale params in (1,10)
def image_scale(img, params):
    params = [params * 0.5 + 1, params * 0.5 + 1]

    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    return res
#(3)Shear params in (1,10)
def image_shear(img, params):
    params = 0.1 * params
    rows, cols, ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


#(4)ratation params in (1,10)
def image_rotation(img, params):
    params = params * 3
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


#(5)contrast params in (1,10)
def image_contrast(img, params):
    params = 1 + params * 0.2
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

    return new_img


#(6)brightness params in (1,10)
def image_brightness(img, params):
    params = params * 10
    beta = params
    new_img = cv2.add(img, beta)                                  # new_img = img*alpha + beta

    return new_img


#(7)blur params in (1,10)
def image_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def precessImag(index, params,img):

    if index == 1:#仿射变换
        imag = image_translation(img, params)
    elif index == 2:#仿射变换
        imag = image_scale(img, params)
    elif index == 3:#仿射变换
        imag = image_shear(img, params)
    elif index == 4:#仿射变换
        imag = image_rotation(img, params)
    elif index == 5:#线性变换
        imag = image_contrast(img, params)
    elif index == 6:#线性变换
        imag = image_brightness(img, params)
    else:#卷积变换
        imag = image_blur(img, params)

    return imag


def generate_pattern(model, x_t, suspicious_indices, step_size, d):
    x_test = x_t.copy()
    layer_out = model.get_layer(index=7).output
    for i in range(len(x_test)):
        print(i)
        for neuron in suspicious_indices:
            neuron = int(neuron)
            loss = tf.reduce_mean(layer_out[...,neuron])
            grads = K.gradients(loss, model.input)[0]
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
            iterate = K.function([model.input], [loss, grads])
            for j in range(10):
                loss_value, grads_value = iterate([np.expand_dims(x_test[i], axis=0)])
                x_test[i] += grads_value[0] * step_size
    return x_test


#可训练层
def get_trainable_layers(model):

    trainable_layers = []
    for i, layer in enumerate(model.layers):
        if "convolutional" in str(layer) or "Dense" in str(layer):
            trainable_layers.append(i)
    return trainable_layers[:-1]

def preprocess_image(img_path, target_size=(192, 256)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = rescale_intensity(input_img_data, in_range=(-255, 255), out_range=(0, 255))
    input_img_data = np.array(input_img_data, dtype=np.uint8)  # to replicate initial model
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data

def getSmallData(list1):
    # random.seed(123)

    random.shuffle(list1)
    # k = int(0.98*len(list1))
    return list1[:50000]


def generate_data(datasetPath):
    fileList = []
    #图像路径存储到fileList
    for file in sorted(os.listdir(datasetPath)):
        if file.endswith(".jpg") or file.endswith(".png"):
            fileList.append(file)
    test = []
    fileList = getSmallData(fileList)#获取测试数据
    for i in range(len(fileList)):
        path = os.path.join(datasetPath, fileList[i])
        test.append(preprocess_image(path))
        print(i)
    test = np.array(test)
    test = test.reshape((test.shape[0], 192, 256, 3))
    return test

# 改
def generateData_jpg(datasetPath):
    fileList = []
    #图像路径存储到fileList
    for file in sorted(os.listdir(datasetPath)):
        if file.endswith(".jpg"):
            fileList.append(file)
    test = []
    #fileList = getSmallData(fileList)#获取500个数据
    for i in range(len(fileList)):
        path = os.path.join(datasetPath, fileList[i])
        test.append(preprocess_image(path))
        print(i)
    test = np.array(test)
    test = test.reshape((test.shape[0], 192, 256, 3))
    return test


# 改
def generateData_png(datasetPath):
    fileList = []
    #图像路径存储到fileList
    for file in sorted(os.listdir(datasetPath)):
        if file.endswith(".png"):
            fileList.append(file)
    test = []
    #fileList = getSmallData(fileList)#获取500个数据
    for i in range(len(fileList)):
        path = os.path.join(datasetPath, fileList[i])
        test.append(preprocess_image(path))
        print(i)
    test = np.array(test)
    test = test.reshape((test.shape[0], 192, 256, 3))
    return test


def generate_label(path):
    temp = np.loadtxt(path, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 0])
    label = []
    for i in range(len(names)):
        n = names[i]
        label.append(float(temp[i, 1]))
    # label = getSmallData(label) #获取测试集标签
    label = np.array(label)
    return label

def generate_testcase_label1(path):
    temp = np.loadtxt(path, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 2])
    label = []
    for i in range(len(names)):
        n = names[i]
        label.append(float(temp[i, 3]))
    #label = getSmallData(label)  # 获取500个标签
    label = np.array(label)
    return label

def generate_testcase_label2(path):
    temp = np.loadtxt(path, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 2])
    label = []
    for i in range(len(names)):
        n = names[i]
        label.append(float(temp[i, 4]))
    #label = getSmallData(label)  # 获取500个标签
    label = np.array(label)
    return label


##生成频谱
def generateSpectrum(output, path, y_label, y_pred, errorRange):
    m = output.shape[0]  # 测试用例数
    n = output.shape[-1]#神经元个数
    spectrum = np.zeros((n, 4))

    ##四元组
    for i in range(m):
        s = abs(y_pred[i] - y_label[i])
        for j in  range(n):
            if np.mean(output[i][...,j]) > 0 and s > errorRange:
                spectrum[j][0] +=1
            if np.mean(output[i][...,j]) > 0 and s <= errorRange:
                spectrum[j][1] += 1
            if np.mean(output[i][...,j]) <= 0 and s > errorRange:
                spectrum[j][2] += 1
            if np.mean(output[i][...,j]) <= 0 and s <= errorRange:
                spectrum[j][3] += 1
    np.savetxt(path, spectrum, fmt='%d')


def generateSpectrums(outputs, errRange):
    for i in range(len(outputs)):
        path = "layer"+str(i+1)
        generateSpectrum(outputs[i], path, errRange)

# 改
def TarantulaError(spectrum):
    Tarantula = np.zeros((len(spectrum),))
    for i in range(len(spectrum)):
        Tarantula[i] = spectrum[i][0] / (spectrum[i][0] + spectrum[i][2]) / (
                    spectrum[i][0] / (spectrum[i][0] + spectrum[i][2])
                    + spectrum[i][1] / (spectrum[i][1] + spectrum[i][3]))
    for i in range(len(spectrum)):
        if np.isnan(Tarantula[i]):
            Tarantula[i] = 99999
    return Tarantula

# 改
def OchiaiError(spectrum):
    Och = np.zeros((len(spectrum),))
    for i in range(len(spectrum)):
        if (spectrum[i][0]+spectrum[i][2])*(spectrum[i][1]+spectrum[i][0]) == 0:
            Och[i] = 99999
        else:
            Och[i] = spectrum[i][0]**1.90/(((spectrum[i][0]+spectrum[i][2])*(spectrum[i][1]+spectrum[i][0]))**0.5)

    return Och

# 改
def D_star(spectrum):
    D = np.zeros((len(spectrum),))
    for i in range(len(spectrum)):
        if spectrum[i][1]+spectrum[i][2] == 0:
            D[i] = 99999
        else:
            D[i] = spectrum[i][0]**3/(spectrum[i][1]+spectrum[i][2])

    return D

def compNums(nums1, nums2):
    a = nums1.reshape((-1,))
    b = nums2.reshape((-1,))
    count = 0
    for i in range(len(b)):
        for j in range(len(a)):
            if a[i] == b[j]:
                count += 1


# 神经元覆盖率
def neuronCover(output):
    m = len(output) # 测试用例
    n = len(output[0]) # 神经元数目
    activate = np.zeros((n,)) # 神经元激活情况
    for i in range (n):
        for j in range(m):
            if output[j][i]>0:
                activate[i] = 1
                break

    return np.sum(activate>0)/n, activate


# K-多节神经元覆盖率
def KMNCov(weights, k,min_max_file):
    min_max = np.load(min_max_file)
    print(min_max)

    N = len(weights[0])
    sum_ = 0
    for i in range(N):
        min_, max_ = min_max[i][0],min_max[i][1]
        sum_ += MNCov(weights[..., i], k, min_,max_)
        print(i)
    return sum_ / (k * N)


def MNCov(weights, k,min_,max_):
    visited = np.zeros((k,))
    n = len(weights)
    differ = max_ - min_
    if differ == 0:
        return k
    s = differ / (k - 1)
    for i in range(n):
        if weights[i]>=min_ and weights[i]<=max_:
            index = int((weights[i] - min_)/s)
            visited[index] += 1
    c = np.sum(visited >= 1)
    return c


# 神经元边界覆盖率
def NBCov(output,min_max_file):
    # 加载保存的每个神经元的最大和最小值
    min_max = np.load(min_max_file)
    n = len(output[0])#神经元数目
    visitUpper = np.zeros(n)
    visitLower = np.zeros(n)
    for i in range(len(output)):#测试用例
        for j in range(len(output[0])):#神经元
            min_, max_ = min_max[j][0], min_max[j][1]
            if output[i][j] > max_:
                visitUpper[j] = 1
            elif output[i][j] < min_:
                visitLower[j] = 1
    Upper = np.sum(visitUpper >= 1)
    Lower = np.sum(visitLower >= 1)
    # print(Upper, Lower)

    return (Upper + Lower) / (2 * n), Upper


# 强神经元激活覆盖率
def SNACov(output, Upper):
    n = len(output[0])

    return Upper / n


# top-k神经元覆盖率
def TKNsCov(output, k):
    # 50,512
    n = len(output[0])
    visited = np.zeros((n,))
    for i in range(len(output)):#测试用例
        visit = np.argsort(output[i])
        for j in range(k):
            visited[visit[-j - 1]] = 1
    sum_ = np.sum(visited >= 1)

    return sum_ ,n

def TKNCov(model, x_test,k):
    trainableLayers = get_trainable_layers(model)
    sums = 0
    ns = 0
    for l in trainableLayers:
        str1 = str(model.layers[int(l)])
        print(str1)
        if "Dense" not in str1 and "convolutional" not in str1:
            continue
        sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=l).output)
        layer_output = sub_model.predict(x_test)
        if layer_output.ndim > 2:
            layer_output = deal_cover_conv(layer_output)
        print(layer_output.shape)
        # Cov2D_output.append(output)
        sum_, n = TKNsCov(layer_output, k)
        sums += sum_
        ns += n
    # covTok = math.tanh(sums / ns * 4)
    covTok = sums / ns
    return covTok
