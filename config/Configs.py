from enum import Enum


class Config(object):
    # 文件目录结构
    log_dir = "log"
    tensorboard_log_dir = "log/tensorboard"
    weight_dir = "log/weight"
    checkpoint_dir = "log/checkpoint"

    model_dir = "model"
    model_output_dir = "model/image"

    other_dir = "other"
    font_dir = "other/font"

    prior_box_dir = "other/prior_box"

    test_data_dir = "data/test"

    train_annotation_path = "data/train_annotation.txt"
    test_annotation_path = "data/test_annotation.txt"
    valid_annotation_path = "data/valid_annotation.txt"

    input_dim = None
    input_source_layers = None
    input_source_layer_width = None
    s_min = None
    s_max = None
    aspect_ratios_per_layer = None
    class_num = None
    bg_class = None
    variances = None
    nms_threshold = None
    softmax_threshold = None
    top_k = None
    alpha = None
    neg_pos_ratio = None
    negatives_for_hard = None
    min_size = None
    max_size = None
    priors = None


class PMethod(Enum):
    Zoom = 0
    Reshape = 1
