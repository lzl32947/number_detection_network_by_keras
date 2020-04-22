from keras import Model
from keras.layers import *

from config.Configs import ModelConfig, Config
from model.ssd_layers import ssd_skeleton


def VGG16_base(inputs):
    output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same',
                    name='conv1_1', )(inputs)
    output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv1_2', )(output)
    output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1', )(output)
    output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv2_1', )(output)
    output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv2_2', )(output)
    output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2', )(output)
    output = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv3_1', )(output)
    output = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv3_2', )(output)
    output = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv3_3', )(output)
    output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3', )(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same",
                    name="conv4_1", )(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same",
                    name="conv4_2", )(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name="conv4_3", )(output)
    output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4', )(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv5_1', )(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv5_2', )(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='conv5_3', )(output)
    output = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5', )(output)

    output = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(6, 6), activation='relu',
                    padding='same', name='fc6')(output)
    output = Conv2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same', name='fc7')(
        output)
    output = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same', name='conv6_1')(
        output)
    output = ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv6_padding")(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid',
                    name="conv6_2")(output)
    output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same", name="conv7_1")(
        output)
    output = ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv7_padding")(output)
    output = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu",
                    name="conv7_2")(output)
    output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same", name="conv8_1")(
        output)
    output = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid",
                    name="conv8_2")(output)
    output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation="relu", padding="same", name="conv9_1")(
        output)
    output = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid",
                    name="conv9_2")(output)
    return output


def VGG16_SSD(prior_box_list):
    model = ModelConfig.VGG16.value
    inputs = Input(shape=(model.input_dim, model.input_dim, 3))
    output = VGG16_base(inputs)
    base_model = Model(inputs=inputs, outputs=output)
    output = ssd_skeleton(base_model, model.input_source_layer_sequence, model.input_source_layer_normalization,
                          prior_box_list, model.aspect_ratios_per_layer, Config.class_names)
    ssd_vgg16 = Model(inputs=inputs, outputs=output)
    return ssd_vgg16
