from keras import Model
from keras.layers import *
from keras.applications import ResNet50
from keras.utils import plot_model

from config.Configs import ModelConfig, Config
from model.ssd_layers import ssd_skeleton


def resnet50_base(inputs):
    resnet50_model = ResNet50(input_tensor=inputs, include_top=False)
    output = resnet50_model.outputs[-1]
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


def ResNet50_SSD(prior_box_list):
    model = ModelConfig.ResNet50.value
    inputs = Input(shape=(model.input_dim, model.input_dim, 3))
    output = resnet50_base(inputs)
    base_model = Model(inputs=inputs, outputs=output)
    output = ssd_skeleton(base_model, model.input_source_layer_sequence, model.input_source_layer_normalization,
                          prior_box_list, model.aspect_ratios_per_layer, Config.class_names)
    ssd_resnet50 = Model(inputs=inputs, outputs=output)
    return ssd_resnet50


if __name__ == '__main__':
    model = ResNet50()
    count = 0
    for i in model.layers:
        print("{}\t->\t{}".format(count, i.name))
        count += 1
