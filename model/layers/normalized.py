import keras
import tensorflow as tf
from keras.layers import Layer
import numpy as np


class L2Normalize(Layer):
    """
    This layer mainly change the visual sight of the image, used in VGG16-based SSD
    """
    def __init__(self, scale, **kwargs):
        self.gamma = None
        self.scale = scale
        self.channel = 3
        super(L2Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        # 获得channel的通道，得到filter的数量
        channels = input_shape[self.channel]
        # 生成可训练的权重
        gamma = self.scale * np.ones(shape=(channels,))
        self.gamma = tf.Variable(gamma, name='{}_gamma'.format(self.name), dtype="float32")
        self.trainable_weights = [self.gamma]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = tf.nn.l2_normalize(inputs, self.channel)
        output = tf.multiply(self.gamma, output)
        return output
