from keras import Model
from keras.layers import *
from keras.utils import plot_model

from networks.vgg16 import vgg16_upper_layers


def ssd_straight_layers():
    """
    该函数返回一个字典对象，包含vgg16网络的直线网络结果
    :return: dict: net
    """
    net = vgg16_upper_layers()
    # shape=(19,19,512)
    net['fc6'] = Conv2D(filters=1024,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        dilation_rate=(6, 6),
                        activation='relu',
                        padding='same',
                        name='fc6')(net['pool5'])
    # shape=(19,19,1024)
    # attention: this layer is not used.
    net['drop6'] = Dropout(rate=0.5, name='drop6')(net['fc6'])
    # shape=(19,19,1024)
    net["fc7"] = Conv2D(filters=1024,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        activation='relu',
                        padding='same',
                        name='fc7')(net['fc6'])
    # shape=(19,19,1024)
    # attention: this layer is not used.
    net['drop7'] = Dropout(rate=0.5, name='drop7')(net['fc7'])
    # shape=(19,19,1024)
    net['conv6_1'] = Conv2D(filters=256,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv6_1')(net["fc7"])
    # shape=(19,19,256)
    net['conv6_padding'] = ZeroPadding2D(padding=((1, 1), (1, 1)),
                                         name="conv6_padding")(net['conv6_1'])
    # shape=(21,21,256)
    # attention: stride and padding are different from the upper ones.
    net['conv6_2'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            activation='relu',
                            padding='valid',
                            name="conv6_2")(net['conv6_padding'])
    # shape=(10,10,512)
    # remember the formula: output = |(input shape - kernel size) / stride| + 1
    net['conv7_1'] = Conv2D(filters=128,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv7_1")(net["conv6_2"])
    # shape=(10,10,128)
    net["conv7_padding"] = ZeroPadding2D(padding=((1, 1), (1, 1)),
                                         name="conv7_padding")(net['conv7_1'])
    # shape=(12,12,128)
    net["conv7_2"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding="valid",
                            activation="relu",
                            name="conv7_2")(net['conv7_padding'])
    # shape=(5,5,256)
    net["conv8_1"] = Conv2D(filters=128,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv8_1")(net["conv7_2"])
    # shape=(5,5,128)
    net["conv8_2"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="valid",
                            name="conv8_2")(net["conv8_1"])
    # shape=(3,3,256)
    net["conv9_1"] = Conv2D(filters=128,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv9_1")(net["conv8_2"])
    # shape=(3,3,128)
    net["conv9_2"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="valid",
                            name="conv9_2")(net["conv9_1"])
    # shape=(1,1,256)
    return net


if __name__ == '__main__':
    nets = ssd_straight_layers()
    model = Model(inputs=nets["input"], outputs=nets["conv9_2"])
    plot_model(model, show_shapes=True, show_layer_names=True)
