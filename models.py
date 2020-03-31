import os
import pickle
import tensorflow as tf
import keras.backend as K

from config.Configs import Config
from model.ResNet50_base import SSD_ResNet50
from model.vgg16_base import SSD_VGG16


def init_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    K.set_session(sess)


def load_config(config_name, cls):
    if config_name == "vgg16":
        if cls == 10:
            from config.VGG16_10 import VGG16_10
            Config.input_dim = VGG16_10.input_dim
            Config.input_source_layers = VGG16_10.input_source_layers
            Config.input_source_layer_width = VGG16_10.input_source_layer_width
            Config.s_min = VGG16_10.s_min
            Config.s_max = VGG16_10.s_max
            Config.aspect_ratios_per_layer = VGG16_10.aspect_ratios_per_layer
            Config.class_num = VGG16_10.class_num
            Config.bg_class = VGG16_10.bg_class
            Config.variances = VGG16_10.variances
            Config.nms_threshold = VGG16_10.nms_threshold
            Config.softmax_threshold = VGG16_10.softmax_threshold
            Config.top_k = VGG16_10.top_k
            Config.alpha = VGG16_10.alpha
            Config.neg_pos_ratio = VGG16_10.neg_pos_ratio
            Config.negatives_for_hard = VGG16_10.negatives_for_hard
            Config.min_size = VGG16_10.min_size
            Config.max_size = VGG16_10.max_size
            Config.priors = VGG16_10.priors
            Config.class_num = cls + 1
            Config.priors = pickle.load(open(os.path.join(Config.prior_box_dir, 'prior_boxes_ssd300.pkl'), 'rb'))
            del VGG16_10
    if config_name == "resnet50":
        if cls == 10:
            from config.ResNet50_10 import ResNet50_10
            Config.input_dim = ResNet50_10.input_dim
            Config.input_source_layers = ResNet50_10.input_source_layers
            Config.input_source_layer_width = ResNet50_10.input_source_layer_width
            Config.s_min = ResNet50_10.s_min
            Config.s_max = ResNet50_10.s_max
            Config.aspect_ratios_per_layer = ResNet50_10.aspect_ratios_per_layer
            Config.class_num = ResNet50_10.class_num
            Config.bg_class = ResNet50_10.bg_class
            Config.variances = ResNet50_10.variances
            Config.nms_threshold = ResNet50_10.nms_threshold
            Config.softmax_threshold = ResNet50_10.softmax_threshold
            Config.top_k = ResNet50_10.top_k
            Config.alpha = ResNet50_10.alpha
            Config.neg_pos_ratio = ResNet50_10.neg_pos_ratio
            Config.negatives_for_hard = ResNet50_10.negatives_for_hard
            Config.min_size = ResNet50_10.min_size
            Config.max_size = ResNet50_10.max_size
            Config.priors = ResNet50_10.priors
            Config.class_num = cls + 1
            Config.priors = pickle.load(open(os.path.join(Config.prior_box_dir, 'prior_boxes_ssd300.pkl'), 'rb'))
            del ResNet50_10


def SSD_Model(model_name, class_num, weight_file=None):
    init_session()
    if model_name == "vgg16":
        if class_num == 10:
            load_config("vgg16", 10)
            model = SSD_VGG16()
            if weight_file is not None:
                for item in weight_file:
                    model.load_weights(item, skip_mismatch=True, by_name=True)
            return model
    if model_name == "resnet50":
        if class_num == 10:
            load_config("resnet50", 10)
            model = SSD_ResNet50()
            if weight_file is not None:
                for item in weight_file:
                    model.load_weights(item, skip_mismatch=True, by_name=True)
            return model
