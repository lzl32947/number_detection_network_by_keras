from keras import Model
from keras.layers import *

from Configs import Config
from model.layers.normalized import L2Normalize
from model.layers.prior_box_layer import PriorBox


def vgg16_upper_layers():
    """
    该函数返回一个字典对象，包含vgg16网络的pool5层(fc6层之前)的网络结果
    :return: dict: net
    """
    # 初始化网络
    net = {}
    # shape=(300,300,3)
    net["input"] = Input(shape=[300, 300, 3], name="input_1")
    # shape=(300,300,64)
    net['conv1_1'] = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding='same',
                            name='conv1_1',
                            )(net['input'])
    # shape=(300,300,64)
    net['conv1_2'] = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv1_2',
                            )(net['conv1_1'])
    # shape=(300,300,64)
    net['pool1'] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool1',
                                )(net['conv1_2'])
    # shape=(150,150,64)
    net['conv2_1'] = Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv2_1',
                            )(net['pool1'])
    # shape=(150,150,128)
    net['conv2_2'] = Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv2_2',
                            )(net['conv2_1'])
    # shape=(150,150,128)
    net['pool2'] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool2',
                                )(net['conv2_2'])
    # shape=(75,75,128)
    net['conv3_1'] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv3_1',
                            )(net['pool2'])
    # shape=(75,75,256)
    net['conv3_2'] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv3_2',
                            )(net['conv3_1'])
    # shape=(75,75,256)
    net["conv3_3"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv3_3',
                            )(net['conv3_2'])
    # shape=(75,75,256)
    net["pool3"] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool3',
                                )(net["conv3_3"])
    # shape=(38,38,256)

    net["conv4_1"] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv4_1",
                            )(net["pool3"])
    # shape=(38,38,512)
    net["conv4_2"] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv4_2",
                            )(net["conv4_1"])
    # shape=(38,38,512)
    net["conv4_3"] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name="conv4_3",
                            )(net["conv4_2"])
    # shape=(38,38,512)
    net['pool4'] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding='same',
                                name='pool4',
                                )(net["conv4_3"])
    # shape=(19,19,512)
    net['conv5_1'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv5_1',
                            )(net['pool4'])
    # shape=(19,19,512)
    net['conv5_2'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv5_2',
                            )(net['conv5_1'])
    # shape=(19,19,512)
    net['conv5_3'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv5_3',
                            )(net['conv5_2'])
    # shape=(19,19,512)
    # attention: strides turn to (1,1) instead of (2,2)
    net['pool5'] = MaxPooling2D(pool_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                name='pool5',
                                )(net['conv5_3'])
    # shape=(19,19,512)
    return net


def ssd_straight_layers():
    """
    该函数返回一个字典对象，包含vgg16网络的直线网络结果
    :return: dict: net
    """
    net = vgg16_upper_layers()
    # shape=(19,19,512)
    net['fc6'] = Conv2D(filters=1024,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        dilation_rate=(6, 6),
                        activation='relu',
                        padding='same',
                        name='fc6')(net['pool5'])
    # shape=(19,19,1024)
    # attention: this layer is not used.
    net['drop6'] = Dropout(rate=0.5, name='drop6')(net['fc6'])
    # shape=(19,19,1024)
    net["fc7"] = Conv2D(filters=1024,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        activation='relu',
                        padding='same',
                        name='fc7')(net['fc6'])
    # shape=(19,19,1024)
    # attention: this layer is not used.
    net['drop7'] = Dropout(rate=0.5, name='drop7')(net['fc7'])
    # shape=(19,19,1024)
    net['conv6_1'] = Conv2D(filters=256,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name='conv6_1')(net["fc7"])
    # shape=(19,19,256)
    net['conv6_padding'] = ZeroPadding2D(padding=((1, 1), (1, 1)),
                                         name="conv6_padding")(net['conv6_1'])
    # shape=(21,21,256)
    # attention: stride and padding are different from the upper ones.
    net['conv6_2'] = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            activation='relu',
                            padding='valid',
                            name="conv6_2")(net['conv6_padding'])
    # shape=(10,10,512)
    # remember the formula: output = |(input shape - kernel size) / stride| + 1
    net['conv7_1'] = Conv2D(filters=128,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv7_1")(net["conv6_2"])
    # shape=(10,10,128)
    net["conv7_padding"] = ZeroPadding2D(padding=((1, 1), (1, 1)),
                                         name="conv7_padding")(net['conv7_1'])
    # shape=(12,12,128)
    net["conv7_2"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding="valid",
                            activation="relu",
                            name="conv7_2")(net['conv7_padding'])
    # shape=(5,5,256)
    net["conv8_1"] = Conv2D(filters=128,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv8_1")(net["conv7_2"])
    # shape=(5,5,128)
    net["conv8_2"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="valid",
                            name="conv8_2")(net["conv8_1"])
    # shape=(3,3,256)
    net["conv9_1"] = Conv2D(filters=128,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                            name="conv9_1")(net["conv8_2"])
    # shape=(3,3,128)
    net["conv9_2"] = Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation="relu",
                            padding="valid",
                            name="conv9_2")(net["conv9_1"])
    # shape=(1,1,256)
    return net


def ssd_network():
    net = ssd_straight_layers()
    # build the block of conv4_3 and output the loc,conf and prior_box
    net['conv4_3_norm'] = L2Normalize(scale=20, name="conv4_3_norm")(net['conv4_3'])
    prior_box_num = len(Config.aspect_ratios_per_layer[0]) + 1
    net['conv4_3_norm_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          name="conv4_3_norm_mbox_loc")(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name="conv4_3_norm_mbox_loc_flat")(net['conv4_3_norm_mbox_loc'])
    net['conv4_3_norm_mbox_conf'] = Conv2D(filters=Config.class_num * prior_box_num,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding="same",
                                           name="conv4_3_norm_mbox_conf")(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name="conv4_3_norm_mbox_conf_flat")(net['conv4_3_norm_mbox_conf'])
    net['conv4_3_norm_mbox_priorbox'] = PriorBox(img_size=(Config.input_dim, Config.input_dim),
                                                 min_size=Config.min_size[0],
                                                 max_size=Config.max_size[0],
                                                 variances=Config.variances,
                                                 aspect_ratios=Config.aspect_ratios_per_layer[0],
                                                 name='conv4_3_norm_mbox_priorbox')(net['conv4_3_norm'])

    # build the block for fc7
    prior_box_num = len(Config.aspect_ratios_per_layer[1]) + 1
    net['fc7_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 name="fc7_mbox_loc")(net['fc7'])
    net['fc7_mbox_loc_flat'] = Flatten(name="fc7_mbox_loc_flat")(net['fc7_mbox_loc'])
    net['fc7_mbox_conf'] = Conv2D(filters=Config.class_num * prior_box_num,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="same",
                                  name="fc7_mbox_conf")(net['fc7'])
    net['fc7_mbox_conf_flat'] = Flatten(name="fc7_mbox_conf_flat")(net['fc7_mbox_conf'])
    net['fc7_mbox_priorbox'] = PriorBox(img_size=(Config.input_dim, Config.input_dim),
                                        min_size=Config.min_size[1],
                                        max_size=Config.max_size[1],
                                        variances=Config.variances,
                                        aspect_ratios=Config.aspect_ratios_per_layer[1],
                                        name='fc7_mbox_priorbox')(net['fc7'])

    # build the block for conv6_2
    prior_box_num = len(Config.aspect_ratios_per_layer[2]) + 1
    net['conv6_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv6_2_mbox_loc")(net['conv6_2'])
    net['conv6_2_mbox_loc_flat'] = Flatten(name="conv6_2_mbox_loc_flat")(net['conv6_2_mbox_loc'])
    net['conv6_2_mbox_conf'] = Conv2D(filters=Config.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv6_2_mbox_conf")(net['conv6_2'])
    net['conv6_2_mbox_conf_flat'] = Flatten(name="conv6_2_mbox_conf_flat")(net['conv6_2_mbox_conf'])
    net['conv6_2_mbox_priorbox'] = PriorBox(img_size=(Config.input_dim, Config.input_dim),
                                            min_size=Config.min_size[2],
                                            max_size=Config.max_size[2],
                                            variances=Config.variances,
                                            aspect_ratios=Config.aspect_ratios_per_layer[2],
                                            name='conv6_2_mbox_priorbox')(net['conv6_2'])

    # build the block for conv7_2
    prior_box_num = len(Config.aspect_ratios_per_layer[3]) + 1
    net['conv7_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv7_2_mbox_loc")(net['conv7_2'])
    net['conv7_2_mbox_loc_flat'] = Flatten(name="conv7_2_mbox_loc_flat")(net['conv7_2_mbox_loc'])
    net['conv7_2_mbox_conf'] = Conv2D(filters=Config.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv7_2_mbox_conf")(net['conv7_2'])
    net['conv7_2_mbox_conf_flat'] = Flatten(name="conv7_2_mbox_conf_flat")(net['conv7_2_mbox_conf'])
    net['conv7_2_mbox_priorbox'] = PriorBox(img_size=(Config.input_dim, Config.input_dim),
                                            min_size=Config.min_size[3],
                                            max_size=Config.max_size[3],
                                            variances=Config.variances,
                                            aspect_ratios=Config.aspect_ratios_per_layer[3],
                                            name='conv7_2_mbox_priorbox')(net['conv7_2'])

    # build for conv8_2
    prior_box_num = len(Config.aspect_ratios_per_layer[4]) + 1
    net['conv8_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv8_2_mbox_loc")(net['conv8_2'])
    net['conv8_2_mbox_loc_flat'] = Flatten(name="conv8_2_mbox_loc_flat")(net['conv8_2_mbox_loc'])
    net['conv8_2_mbox_conf'] = Conv2D(filters=Config.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv8_2_mbox_conf")(net['conv8_2'])
    net['conv8_2_mbox_conf_flat'] = Flatten(name="conv8_2_mbox_conf_flat")(net['conv8_2_mbox_conf'])
    net['conv8_2_mbox_priorbox'] = PriorBox(img_size=(Config.input_dim, Config.input_dim),
                                            min_size=Config.min_size[4],
                                            max_size=Config.max_size[4],
                                            variances=Config.variances,
                                            aspect_ratios=Config.aspect_ratios_per_layer[4],
                                            name='conv8_2_mbox_priorbox')(net['conv8_2'])

    # build for conv9_2
    prior_box_num = len(Config.aspect_ratios_per_layer[5]) + 1
    net['conv9_2_mbox_loc'] = Conv2D(filters=4 * prior_box_num,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     name="conv9_2_mbox_loc")(net['conv9_2'])
    net['conv9_2_mbox_loc_flat'] = Flatten(name="conv9_2_mbox_loc_flat")(net['conv9_2_mbox_loc'])
    net['conv9_2_mbox_conf'] = Conv2D(filters=Config.class_num * prior_box_num,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name="conv9_2_mbox_conf")(net['conv9_2'])
    net['conv9_2_mbox_conf_flat'] = Flatten(name="conv9_2_mbox_conf_flat")(net['conv9_2_mbox_conf'])
    net['conv9_2_mbox_priorbox'] = PriorBox(img_size=(Config.input_dim, Config.input_dim),
                                            min_size=Config.min_size[5],
                                            max_size=Config.max_size[5],
                                            variances=Config.variances,
                                            aspect_ratios=Config.aspect_ratios_per_layer[5],
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
    net['mbox_conf'] = Reshape((box_num, Config.class_num), name='mbox_conf_logits')(net['mbox_conf'])
    # add softmax function
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    # final get the predict
    net['predictions'] = concatenate([net['mbox_loc'],
                                      net['mbox_conf'],
                                      net['mbox_priorbox']],
                                     axis=2,
                                     name='predictions')
    return net


def SSD_VGG16():
    net = ssd_network()
    model = Model(net['input'], net['predictions'])
    return model
