from keras import Model
from keras.layers import *
from keras.utils import plot_model
from layers.normalized import L2Normalize
from layers.prior_box_layer import PriorBox
from networks.ssd_straight import ssd_straight_layers
from parameter.parameters import HyperParameter, Parameter
import tensorflow as tf


def ssd_network():
    net = ssd_straight_layers()

    # build the block of conv4_3 and output the loc,conf and prior_box
    net['conv4_3_norm'] = L2Normalize(scale=20, name="conv4_3_norm")(net['conv4_3'])
    prior_box_num = len(HyperParameter.aspect_ratios_per_layer[0]) + 1
    net['conv4_3_norm_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          name="conv4_3_norm_mbox_loc")(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name="conv4_3_norm_mbox_loc_flat")(net['conv4_3_norm_mbox_loc'])
    net['conv4_3_norm_mbox_conf'] = Conv2D(filters=HyperParameter.class_num * prior_box_num,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding="same",
                                           name="conv4_3_norm_mbox_conf")(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name="conv4_3_norm_mbox_conf_flat")(net['conv4_3_norm_mbox_conf'])
    net['conv4_3_norm_mbox_priorbox'] = PriorBox(img_size=(HyperParameter.min_dim, HyperParameter.min_dim),
                                                 min_size=Parameter.min_size[0],
                                                 max_size=Parameter.max_size[0],
                                                 variances=HyperParameter.variances,
                                                 aspect_ratios=HyperParameter.aspect_ratios_per_layer[0],
                                                 name='conv4_3_norm_mbox_priorbox')(net['conv4_3_norm'])

    # build the block for fc7
    prior_box_num = len(HyperParameter.aspect_ratios_per_layer[1]) + 1
    net['fc7_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 name="fc7_mbox_loc")(net['fc7'])
    net['fc7_mbox_loc_flat'] = Flatten(name="fc7_mbox_loc_flat")(net['fc7_mbox_loc'])
    net['fc7_mbox_conf'] = Conv2D(filters=HyperParameter.class_num * prior_box_num,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="same",
                                  name="fc7_mbox_conf")(net['fc7'])
    net['fc7_mbox_conf_flat'] = Flatten(name="fc7_mbox_conf_flat")(net['fc7_mbox_conf'])
    net['fc7_mbox_priorbox'] = PriorBox(img_size=(HyperParameter.min_dim, HyperParameter.min_dim),
                                        min_size=Parameter.min_size[1],
                                        max_size=Parameter.max_size[1],
                                        variances=HyperParameter.variances,
                                        aspect_ratios=HyperParameter.aspect_ratios_per_layer[1],
                                        name='fc7_mbox_priorbox')(net['fc7'])

    # build the block for conv6_2
    prior_box_num = len(HyperParameter.aspect_ratios_per_layer[2]) + 1
    net['conv6_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv6_2_mbox_loc")(net['conv6_2'])
    net['conv6_2_mbox_loc_flat'] = Flatten(name="conv6_2_mbox_loc_flat")(net['conv6_2_mbox_loc'])
    net['conv6_2_mbox_conf'] = Conv2D(filters=HyperParameter.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv6_2_mbox_conf")(net['conv6_2'])
    net['conv6_2_mbox_conf_flat'] = Flatten(name="conv6_2_mbox_conf_flat")(net['conv6_2_mbox_conf'])
    net['conv6_2_mbox_priorbox'] = PriorBox(img_size=(HyperParameter.min_dim, HyperParameter.min_dim),
                                            min_size=Parameter.min_size[2],
                                            max_size=Parameter.max_size[2],
                                            variances=HyperParameter.variances,
                                            aspect_ratios=HyperParameter.aspect_ratios_per_layer[2],
                                            name='conv6_2_mbox_priorbox')(net['conv6_2'])

    # build the block for conv7_2
    prior_box_num = len(HyperParameter.aspect_ratios_per_layer[3]) + 1
    net['conv7_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv7_2_mbox_loc")(net['conv7_2'])
    net['conv7_2_mbox_loc_flat'] = Flatten(name="conv7_2_mbox_loc_flat")(net['conv7_2_mbox_loc'])
    net['conv7_2_mbox_conf'] = Conv2D(filters=HyperParameter.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv7_2_mbox_conf")(net['conv7_2'])
    net['conv7_2_mbox_conf_flat'] = Flatten(name="conv7_2_mbox_conf_flat")(net['conv7_2_mbox_conf'])
    net['conv7_2_mbox_priorbox'] = PriorBox(img_size=(HyperParameter.min_dim, HyperParameter.min_dim),
                                            min_size=Parameter.min_size[3],
                                            max_size=Parameter.max_size[3],
                                            variances=HyperParameter.variances,
                                            aspect_ratios=HyperParameter.aspect_ratios_per_layer[3],
                                            name='conv7_2_mbox_priorbox')(net['conv7_2'])

    # build for conv8_2
    prior_box_num = len(HyperParameter.aspect_ratios_per_layer[4]) + 1
    net['conv8_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv8_2_mbox_loc")(net['conv8_2'])
    net['conv8_2_mbox_loc_flat'] = Flatten(name="conv8_2_mbox_loc_flat")(net['conv8_2_mbox_loc'])
    net['conv8_2_mbox_conf'] = Conv2D(filters=HyperParameter.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv8_2_mbox_conf")(net['conv8_2'])
    net['conv8_2_mbox_conf_flat'] = Flatten(name="conv8_2_mbox_conf_flat")(net['conv8_2_mbox_conf'])
    net['conv8_2_mbox_priorbox'] = PriorBox(img_size=(HyperParameter.min_dim, HyperParameter.min_dim),
                                            min_size=Parameter.min_size[4],
                                            max_size=Parameter.max_size[4],
                                            variances=HyperParameter.variances,
                                            aspect_ratios=HyperParameter.aspect_ratios_per_layer[4],
                                            name='conv8_2_mbox_priorbox')(net['conv8_2'])

    # build for conv9_2
    prior_box_num = len(HyperParameter.aspect_ratios_per_layer[5]) + 1
    net['conv9_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv9_2_mbox_loc")(net['conv9_2'])
    net['conv9_2_mbox_loc_flat'] = Flatten(name="conv9_2_mbox_loc_flat")(net['conv9_2_mbox_loc'])
    net['conv9_2_mbox_conf'] = Conv2D(filters=HyperParameter.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv9_2_mbox_conf")(net['conv9_2'])
    net['conv9_2_mbox_conf_flat'] = Flatten(name="conv9_2_mbox_conf_flat")(net['conv9_2_mbox_conf'])
    net['conv9_2_mbox_priorbox'] = PriorBox(img_size=(HyperParameter.min_dim, HyperParameter.min_dim),
                                            min_size=Parameter.min_size[5],
                                            max_size=Parameter.max_size[5],
                                            variances=HyperParameter.variances,
                                            aspect_ratios=HyperParameter.aspect_ratios_per_layer[5],
                                            name='conv9_2_mbox_priorbox')(net['conv9_2'])

    # concatenate all necessary result
    net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                                   net['fc7_mbox_loc_flat'],
                                   net['conv6_2_mbox_loc_flat'],
                                   net['conv7_2_mbox_loc_flat'],
                                   net['conv8_2_mbox_loc_flat'],
                                   net['conv9_2_mbox_loc_flat']],
                                  axis=1, name='mbox_loc')

    net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                                    net['fc7_mbox_conf_flat'],
                                    net['conv6_2_mbox_conf_flat'],
                                    net['conv7_2_mbox_conf_flat'],
                                    net['conv8_2_mbox_conf_flat'],
                                    net['conv9_2_mbox_conf_flat']],
                                   axis=1, name='mbox_conf')

    net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                        net['fc7_mbox_priorbox'],
                                        net['conv6_2_mbox_priorbox'],
                                        net['conv7_2_mbox_priorbox'],
                                        net['conv8_2_mbox_priorbox'],
                                        net['conv9_2_mbox_priorbox']],
                                       axis=1, name='mbox_priorbox')
    # get the total box num and divided by 4 to get the tuple,
    # and one tuple should have center_x,center_y,width,height(not sorted)
    box_num = net['mbox_loc']._keras_shape[-1] // 4
    # reshape the loc to (?,4)
    net['mbox_loc'] = Reshape((box_num, 4), name='mbox_loc_final')(net['mbox_loc'])
    # also reshape to make classification
    net['mbox_conf'] = Reshape((box_num, HyperParameter.class_num), name='mbox_conf_logits')(net['mbox_conf'])
    # add softmax function
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    # final get the predict
    net['predictions'] = concatenate([net['mbox_loc'],
                                      net['mbox_conf'],
                                      net['mbox_priorbox']],
                                     axis=2,
                                     name='predictions')
    return net


def get_SSD_model():
    net = ssd_network()
    model = Model(net['input'], net['predictions'])
    return model


if __name__ == '__main__':
    models = get_SSD_model()
    plot_model(models, to_file="model_number.png", show_layer_names=True, show_shapes=True)
