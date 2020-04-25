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

    test_data_dir = "data/test"

    train_annotation_path = "data/train_annotation.txt"
    test_annotation_path = "data/test_annotation.txt"
    valid_annotation_path = "data/valid_annotation.txt"

    single_digits_dir = "data/single_digits"

    detection_result_dir = "data/result"

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    bg_class = 0

    neg_pos_ratio = 3.0
    negatives_for_hard = 100.0
    alpha = 1.0


class model_config(Config):
    def __init__(self,
                 input_dim,
                 input_source_layer_sequence,
                 input_source_layer_normalization,
                 input_source_layers_name,
                 input_source_layer_width,
                 s_min,
                 s_max,
                 aspect_ratios_per_layer,
                 variances,
                 nms_threshold,
                 softmax_threshold,
                 top_k,
                 min_size,
                 max_size,
                 ):
        super().__init__()
        self._input_dim = input_dim
        self._input_source_layers_name = input_source_layers_name
        self._input_source_layer_normalization = input_source_layer_normalization
        self._input_source_layer_sequence = input_source_layer_sequence
        self._input_source_layer_width = input_source_layer_width
        self._s_min = s_min
        self._s_max = s_max
        self._aspect_ratios_per_layer = aspect_ratios_per_layer
        self._variances = variances
        self._nms_threshold = nms_threshold
        self._softmax_threshold = softmax_threshold
        self._top_k = top_k
        self._min_size = min_size
        self._max_size = max_size

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def input_source_layers(self):
        return self._input_source_layers_name

    @property
    def input_source_layer_normalization(self):
        return self._input_source_layer_normalization

    @property
    def input_source_layer_sequence(self):
        return self._input_source_layer_sequence

    @property
    def input_source_layer_width(self):
        return self._input_source_layer_width

    @property
    def s_min(self):
        return self._s_min

    @property
    def s_max(self):
        return self._s_max

    @property
    def aspect_ratios_per_layer(self):
        return self._aspect_ratios_per_layer

    @property
    def variances(self):
        return self._variances

    @property
    def nms_threshold(self):
        return self._nms_threshold

    @property
    def softmax_threshold(self):
        return self._softmax_threshold

    @property
    def top_k(self):
        return self._top_k

    @property
    def min_size(self):
        return self._min_size

    @property
    def max_size(self):
        return self._max_size


class ModelConfig(Enum):
    VGG16 = model_config(
        input_dim=300,
        input_source_layer_sequence=[13, 20, 23, 26, 28, 30],
        input_source_layer_normalization=[True, False, False, False, False, False],
        input_source_layers_name=['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2'],
        input_source_layer_width=[38, 19, 10, 5, 3, 1],
        s_min=20,
        s_max=90,
        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5]],
        variances=[0.1, 0.1, 0.2, 0.2],
        nms_threshold=0.5,
        softmax_threshold=0.5,
        top_k=400,
        min_size=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
        max_size=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
    )
    ResNet50 = model_config(
        input_dim=300,
        input_source_layer_sequence=[80, 142, 174, 177, 179, 181],
        input_source_layer_normalization=[True, False, False, False, False, False],
        input_source_layers_name=['activation_22', 'activation_40', 'activation_49', 'conv7_2', 'conv8_2', 'conv9_2'],
        input_source_layer_width=[38, 19, 10, 5, 3, 1],
        s_min=20,
        s_max=90,
        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5]],
        variances=[0.1, 0.1, 0.2, 0.2],
        nms_threshold=0.5,
        softmax_threshold=0.5,
        top_k=400,
        min_size=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
        max_size=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
    )
    ResNet101 = model_config(
        input_dim=300,
        input_source_layer_sequence=[80, 312, 344, 347, 349, 351],
        input_source_layer_normalization=[True, False, False, False, False, False],
        input_source_layers_name=['conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out', 'conv7_2', 'conv8_2', 'conv9_2'],
        input_source_layer_width=[38, 19, 10, 5, 3, 1],
        s_min=20,
        s_max=90,
        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5]],
        variances=[0.1, 0.1, 0.2, 0.2],
        nms_threshold=0.5,
        softmax_threshold=0.5,
        top_k=400,
        min_size=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
        max_size=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
    )
    MobileNetV2 = model_config(
        input_dim=300,
        input_source_layer_sequence=[57, 119, 154, 157, 159, 161],
        input_source_layer_normalization=[True, False, False, False, False, False],
        input_source_layers_name=['block_6_expand_relu', 'block_13_expand_relu', 'out_relu', 'conv7_2', 'conv8_2', 'conv9_2'],
        input_source_layer_width=[38, 19, 10, 5, 3, 1],
        s_min=20,
        s_max=90,
        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                 [1.0, 2.0, 0.5],
                                 [1.0, 2.0, 0.5]],
        variances=[0.1, 0.1, 0.2, 0.2],
        nms_threshold=0.5,
        softmax_threshold=0.5,
        top_k=400,
        min_size=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
        max_size=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
    )


class PMethod(Enum):
    Reshape = 1
    Zoom = 2
