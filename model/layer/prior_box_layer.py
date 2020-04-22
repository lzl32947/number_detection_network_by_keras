from keras.layers import Layer
import numpy as np
import tensorflow as tf


class PriorBox(Layer):
    def __init__(self, output_result, **kwargs):
        self.output_result = output_result
        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_len = input_shape[0]
        return input_len, self.output_result.shape[0], self.output_result.shape[1]

    def call(self, x, **kwargs):
        outputs = tf.Variable(self.output_result, dtype="float32")
        outputs = tf.expand_dims(outputs, 0)
        pattern = [tf.shape(x)[0], 1, 1]
        outputs = tf.tile(outputs, pattern)
        return outputs
