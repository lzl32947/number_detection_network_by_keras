from keras.layers import Layer
import numpy as np
import tensorflow as tf


class PriorBox(Layer):
    # This layer is just for calculating out the frame of the prior box.
    # and the layer is not trainable.
    # 该层只是提供了先验框的边框，并且不会被训练
    # 最终输出的形状为先验框左上角和右下角对应的坐标点，以及添加的variances
    # 输出形状为(?,8),其中后一维度的前4为坐标点，后4为添加的variances
    def __init__(self, img_size, min_size, max_size, variances, aspect_ratios, **kwargs):
        self.height = img_size[0]
        self.width = img_size[1]
        self.min_size = min_size
        self.max_size = max_size
        self.variances = variances
        self.aspect_ratios = aspect_ratios
        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """
        custom the output shape of the layer
        :param input_shape: the feature map
        :return: a tensor which has shape (input_shape[0],prior_box_num,8)
        """
        input_len = input_shape[0]
        layer_height = input_shape[1]
        layer_width = input_shape[2]
        prior_box_num = len(self.aspect_ratios) + 1
        output_len = 8
        return input_len, layer_height * layer_width * prior_box_num, output_len

    def call(self, x, **kwargs):
        """
        :param x: the feature map of the input layer, a tensor
        :param kwargs: not used
        """
        input_shape = x.get_shape().as_list()
        # input_shape should have the shape like (?,layer_height,layer_width,filters)
        input_height = input_shape[1]
        input_width = input_shape[2]
        box_width = []
        box_height = []
        # add the box into list
        for i in self.aspect_ratios:
            if i == 1.0:
                # add the two square
                box_height.append(self.min_size)
                box_width.append(self.min_size)
                box_height.append(np.sqrt(self.min_size * self.max_size))
                box_width.append(np.sqrt(self.min_size * self.max_size))
            else:
                box_width.append(np.sqrt(i) * self.min_size)
                box_height.append(1 / np.sqrt(i) * self.min_size)
        box_width_half = [0.5 * w_l for w_l in box_width]
        box_height_half = [0.5 * h_l for h_l in box_height]
        # calculate the center
        # the origin img shape in (self.height,self.width)
        # the layer input shape in (input_height,input_width)
        step_y = self.height / input_height
        step_x = self.width / input_width
        x_distribution = np.linspace(0.5 * step_x, self.width - 0.5 * step_x, input_width)
        y_distribution = np.linspace(0.5 * step_y, self.height - 0.5 * step_y, input_height)
        c_x, c_y = np.meshgrid(x_distribution, y_distribution)
        # c_x and c_y with shape(input_height,input_width)
        # then flatten the c_x and c_y to make them match each other
        c_x = np.reshape(c_x, (-1, 1))
        c_y = np.reshape(c_y, (-1, 1))
        # then concatenate them vertically
        center_list = np.concatenate((c_x, c_y), axis=1)
        # center_list should have shape (input_height * input_width,2)
        # to get the count of prior box
        num_prior_box = len(self.aspect_ratios) + 1
        # to generate the left-top and the right-bottom of the rectangle
        # 计算prior box的左上角和右下角作为坐标
        # 计算后的shape should be (input_width * input_height,4 * num_prior_box)
        # 因为一个prior box 需要4个参数定位
        # 因此需要将center_list的第二维度扩展2倍，到4
        output_list = np.tile(center_list, (1, 2 * num_prior_box))
        # 对output_list第二维度的元素做运算：
        # 以4 * prior_box_num为一组，单组形式为:...,x,y,x,y,x,y,x,y...
        output_list[:, ::4] -= box_width_half
        output_list[:, 1::4] -= box_height_half
        output_list[:, 2::4] += box_width_half
        output_list[:, 3::4] += box_height_half
        # 转换为对应于原图的小数形式
        output_list[:, ::2] /= self.width
        output_list[:, 1::2] /= self.height
        # 之后将其按第二维度四个参数为一组reshape
        # 此时每个长方形的最后一维度的四个参数代表prior box的左上角和右下角
        output_list = np.reshape(output_list, (-1, 4))
        # this time output_list should with shape (prior_box_num * input_width * input_height, 4)
        # 归一化output_list使得其在0-1之间
        output_list = np.clip(output_list, 0.0, 1.0)
        # 添加variance到output维度中
        # variance should have shape(4,)
        variances = np.tile(self.variances, (output_list.shape[0], 1))
        # variance now has shape (prior_box_num * input_width * input_height,4)
        outputs = np.concatenate((output_list, variances), axis=1)
        # outputs should have shape (prior_box_num * input_width * input_height,8)
        # 包装变量为tf.tensor，注意初始化时要指定dtype以防出现ValueError
        outputs = tf.Variable(outputs, dtype="float32")
        # 升高维度，添加输入的第一个参数(即input_shape[0])
        outputs = tf.expand_dims(outputs, 0)
        # 对num进行赋值
        pattern = [tf.shape(x)[0], 1, 1]
        outputs = tf.tile(outputs, pattern)
        # 此时输入维度和输出维度相同
        # 即x.shape == output_result.shape
        return outputs
