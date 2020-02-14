import keras
from keras.layers import *


def vgg16_upper_layers():
    """
    该函数返回一个字典对象，包含vgg16网络的pool5层(fc6层之前)的网络结果
    :return: dict: net
    """
    # 初始化网络
    net = {}
    # shape=(300,300,3)
    net["input"] = Input(shape=[300, 300, 3], name="input_1")
    # shape=(300,300,64)
    net['conv1_1'] = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding='same',
                            name='conv1_1')(net['input'])
    # shape=(300,300,64)
    net['conv1_2'] = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv1_2')(net['conv1_1'])
    # shape=(300,300,64)
    net['pool1'] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool1')(net['conv1_2'])
    # shape=(150,150,64)
    net['conv2_1'] = Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv2_1')(net['pool1'])
    # shape=(150,150,128)
    net['conv2_2'] = Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv2_2')(net['conv2_1'])
    # shape=(150,150,128)
    net['pool2'] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool2')(net['conv2_2'])
    # shape=(75,75,128)
    net['conv3_1'] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv3_1')(net['pool2'])
    # shape=(75,75,256)
    net['conv3_2'] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv3_2')(net['conv3_1'])
    # shape=(75,75,256)
    net["conv3_3"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv3_3')(net['conv3_2'])
    # shape=(75,75,256)
    net["pool3"] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool3')(net["conv3_3"])
    # shape=(38,38,256)

    net["conv4_1"] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv4_1")(net["pool3"])
    # shape=(38,38,512)
    net["conv4_2"] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv4_2")(net["conv4_1"])
    # shape=(38,38,512)
    net["conv4_3"] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name="conv4_3")(net["conv4_2"])
    # shape=(38,38,512)
    net['pool4'] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool4')(net["conv4_3"])
    # shape=(19,19,512)
    net['conv5_1'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv5_1')(net['pool4'])
    # shape=(19,19,512)
    net['conv5_2'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv5_2')(net['conv5_1'])
    # shape=(19,19,512)
    net['conv5_3'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv5_3')(net['conv5_2'])
    # shape=(19,19,512)
    # attention: strides turn to (1,1) instead of (2,2)
    net['pool5'] = MaxPooling2D(pool_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                name='pool5')(net['conv5_3'])
    # shape=(19,19,512)
    return net
