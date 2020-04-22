from keras.layers import *

from model.layer.normalized import L2Normalize
from model.layer.prior_box_layer import PriorBox


def get_layer_name(layer):
    return layer.name.split('/')[0]


def ssd_skeleton(model, input_source_layer_sequence, use_normalization, prior_box_list, aspect_ratio, class_names):
    loc_list = []
    conf_list = []
    prior_list = []
    for i in range(0, len(input_source_layer_sequence)):
        layer_id = input_source_layer_sequence[i]
        layer = model.layers[layer_id].output
        if use_normalization[i]:
            layer = L2Normalize(scale=20, name=get_layer_name(layer) + "_norm")(layer)
        prior_box_num = len(aspect_ratio[i]) + 1
        layer_loc = Conv2D(filters=4 * prior_box_num,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding="same",
                           name=get_layer_name(layer) + "_mbox_loc")(layer)
        layer_loc = Flatten(name=get_layer_name(layer_loc) + "_flat")(layer_loc)
        loc_list.append(layer_loc)

        layer_conf = Conv2D(filters=(len(class_names) + 1) * prior_box_num,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            name=get_layer_name(layer) + "_mbox_conf")(layer)
        layer_conf = Flatten(name=get_layer_name(layer_conf) + "_flat")(layer_conf)
        conf_list.append(layer_conf)

        layer_prior_box = PriorBox(prior_box_list[i],
                                   name=get_layer_name(layer) + '_mbox_priorbox')(layer)
        prior_list.append(layer_prior_box)

    loc_layer = concatenate(loc_list, axis=1, name='mbox_loc')
    conf_layer = concatenate(conf_list, axis=1, name='mbox_conf')
    prior_box_layer = concatenate(prior_list, axis=1, name='mbox_priorbox')
    box_num = loc_layer._keras_shape[-1] // 4
    loc_layer = Reshape((box_num, 4), name='mbox_loc_final')(loc_layer)
    conf_layer = Reshape((box_num, len(class_names) + 1), name='mbox_conf_logits')(conf_layer)
    conf_layer = Activation('softmax', name='mbox_conf_final')(conf_layer)

    output = concatenate([loc_layer, conf_layer, prior_box_layer], axis=2, name='predictions')
    return output
