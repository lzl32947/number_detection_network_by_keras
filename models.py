import os
import pickle
import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model

from Configs import Config
from model.vgg16_base import SSD_VGG16


def init_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    K.set_session(sess)


def choose_model(model_name, class_num):
    init_session()
    if model_name == "vgg16":
        Config.class_num = class_num + 1
        Config.priors = pickle.load(open(os.path.join(Config.prior_box_dir, 'prior_boxes_ssd300.pkl'), 'rb'))
        model = SSD_VGG16()
        return model
