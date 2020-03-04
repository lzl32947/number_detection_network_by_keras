import numpy as np


class HyperParameter(object):
    # 输入的最小尺度
    min_dim = 300
    # 特征图的来源
    input_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    # 特征图像的边长
    input_source_layer_width = [38, 19, 10, 5, 3, 1]
    # 最底层先验框的大小(scale):0.2
    s_min = 20
    # 最顶层先验框的大小(scale):0.9
    s_max = 90
    # 定义先验框的大小的超参数
    aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                               [1.0, 2.0, 0.5],
                               [1.0, 2.0, 0.5]]
    # 定义输出的种类(包括一种背景)
    class_num = 11
    # 定义背景类的序号
    bg_class = 0
    # 定义用于加快训练的variance
    variances = [0.1, 0.1, 0.2, 0.2]
    # 定义非极大值抑制的阈值
    threshold = 0.5
    # 定义最后选取的top k个元素的大小
    top_k = 400
    # TODO:Add meaning to this parameter:alpha
    alpha = 1.0
    # 负样本和正样本之间的比例
    neg_pos_ratio = 3.0
    # TODO:Add meaning to this parameter:negatives_for_hard
    negatives_for_hard = 100.0
    # VGG16 Trainable
    vgg16_trainable = True


class Parameter(object):
    # m 为计算步长公式中提取的层数
    m = len(HyperParameter.input_source_layers)
    # 通过s_min和s_max计算出每一个特征层对应的Anchor尺寸
    # 初始化s_list
    s_list = []
    # 得到的s_list为一个存储着scale的列表,归一化成不超过1的小数形式
    step = int(np.floor((HyperParameter.s_max - HyperParameter.s_min) / (m - 2)))
    # 初始化每一层先验框的min_size和max_size大小
    min_size = []
    max_size = []
    # 由于在文件中，conv4_3的先验框大小被单独计算，在此先行写入
    # Attention: this is not a scale!
    # 注意: 这不是一个比例！
    min_size.append(HyperParameter.min_dim * 10 / 100.0)
    max_size.append(HyperParameter.min_dim * 20 / 100.0)
    # 计算剩下几层的先验框大小
    for ratio in range(HyperParameter.s_min, HyperParameter.s_max + 1, step):
        min_size.append(HyperParameter.min_dim * ratio / 100.0)
        max_size.append(HyperParameter.min_dim * (ratio + step) / 100.0)


if __name__ == '__main__':
    # for index in range(0, len(HyperParameter.aspect_ratios_per_layer)):
    #     box_width = []
    #     box_height = []
    #     # add the box into list
    #     for i in HyperParameter.aspect_ratios_per_layer[index]:
    #         if i == 1.0:
    #             # add the two square
    #             box_height.append(Parameter.min_size[index])
    #             box_width.append(Parameter.min_size[index])
    #             box_height.append(np.sqrt(Parameter.min_size[index] * Parameter.max_size[index]))
    #             box_width.append(np.sqrt(Parameter.min_size[index] * Parameter.max_size[index]))
    #         else:
    #             box_width.append(np.sqrt(i) * Parameter.min_size[index])
    #             box_height.append(1 / np.sqrt(i) * Parameter.min_size[index])
    #     print(box_height)
    #     print(box_width)
    #     box_height_half = [0.5 * w_h for w_h in box_height]
    #     box_width_half = [0.5 * w_l for w_l in box_width]
    print(Parameter.max_size)
