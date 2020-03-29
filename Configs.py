from enum import Enum


class Config(object):
    # 文件目录结构
    log_dir = "./log"
    tensorboard_log_dir = "./log/tensorboard"
    weight_dir = "./log/weight"
    checkpoint_dir = "./log/checkpoint"

    model_dir = "./model"
    model_output_dir = "./model/image"

    other_dir = "./other"
    font_dir = "./other/font"

    prior_box_dir = "./other/prior_box"

    test_data_dir = "./data"

    # 输入的最小尺度
    input_dim = 300
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
    class_num = 21
    # 定义背景类的序号
    bg_class = 0
    # 定义用于加快训练的variance
    variances = [0.1, 0.1, 0.2, 0.2]
    # 定义非极大值抑制的阈值
    nms_threshold = 0.5
    # 定义softmax的阈值
    softmax_threshold = 0.5
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
    # max_size & min_size
    min_size = [30.0, 60.0, 111.0, 162.0, 213.0, 264.0]
    max_size = [60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
    # prior_boxes
    priors = None


class PMethod(Enum):
    Zoom = 0
    Reshape = 1


