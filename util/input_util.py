import os
import random
import tensorflow as tf
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image

from config.Configs import PMethod, Config
from util.image_generator import get_image_number_list, generate_single_image
from util.image_util import zoom_image, resize_image


def calculate_max_iou(pos, prior_box):
    """
    Return the IoU of the single box with anchor
    :param pos: numpy array, typically with shape (1,4)
    :param prior_box: numpy array
    :return: numpy array, IoU result
    """
    pri = prior_box
    _pos = np.expand_dims(pos, axis=0).repeat(len(pri), axis=0)
    inter_width = np.minimum(_pos[:, 2], pri[:, 2]) - np.maximum(_pos[:, 0], pri[:, 0])
    inter_height = np.minimum(_pos[:, 3], pri[:, 3]) - np.maximum(_pos[:, 1], pri[:, 1])
    inter_area = np.maximum(0, inter_height) * np.maximum(0, inter_width)
    full_area = (_pos[:, 3] - _pos[:, 1]) * (_pos[:, 2] - _pos[:, 0]) + (pri[:, 3] - pri[:, 1]) * (
            pri[:, 2] - pri[:, 0])
    iou = inter_area / (full_area - inter_area)
    return np.argmax(iou)


def encode_box(box, anchor, variance):
    """
    Encode the single box to offset format.
    :param variance: tuple, variance used, with shape (4,)
    :param box: numpy array, the real box with format (x_min,y_min,x_max,y_max)
    :param anchor: numpy array
    :return: the encoded box
    """
    result = np.zeros(shape=(4,), dtype=np.float)
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]

    anchor_center = 0.5 * (anchor[:2] + anchor[2:4])
    anchor_wh = (anchor[2:4] - anchor[:2])

    result[:2] = box_center - anchor_center
    result[:2] /= anchor_wh

    result[2:4] = np.log(box_wh / anchor_wh)
    result[0] *= variance[0]
    result[1] *= variance[1]
    result[2] *= variance[2]
    result[3] *= variance[3]

    return result


def get_prior_box_list(model_name):
    model = model_name.value
    result = []
    input_dim = model.input_dim
    variances = model.variances
    for feature_map_size, aspect_ratio, min_size, max_size in zip(model.input_source_layer_width,
                                                                  model.aspect_ratios_per_layer, model.min_size,
                                                                  model.max_size):
        box_width = []
        box_height = []
        for a in aspect_ratio:
            if a == 1.0:
                box_height.append(min_size)
                box_width.append(max_size)
                box_height.append(np.sqrt(min_size * max_size))
                box_width.append(np.sqrt(min_size * max_size))
            else:
                box_width.append(np.sqrt(a) * min_size)
                box_height.append(1 / np.sqrt(a) * min_size)
        box_width_half = [0.5 * w_l for w_l in box_width]
        box_height_half = [0.5 * h_l for h_l in box_height]
        step_y = input_dim / feature_map_size
        step_x = input_dim / feature_map_size
        x_distribution = np.linspace(0.5 * step_x, input_dim - 0.5 * step_x, feature_map_size)
        y_distribution = np.linspace(0.5 * step_y, input_dim - 0.5 * step_y, feature_map_size)
        c_x, c_y = np.meshgrid(x_distribution, y_distribution)
        c_x = np.reshape(c_x, (-1, 1))
        c_y = np.reshape(c_y, (-1, 1))
        center_list = np.concatenate((c_x, c_y), axis=1)
        num_prior_box = len(aspect_ratio) + 1
        output_list = np.tile(center_list, (1, 2 * num_prior_box))
        output_list[:, ::4] -= box_width_half
        output_list[:, 1::4] -= box_height_half
        output_list[:, 2::4] += box_width_half
        output_list[:, 3::4] += box_height_half
        output_list[:, ::2] /= input_dim
        output_list[:, 1::2] /= input_dim
        output_list = np.reshape(output_list, (-1, 4))
        output_list = np.clip(output_list, 0.0, 1.0)
        variance = np.tile(variances, (output_list.shape[0], 1))
        outputs = np.concatenate((output_list, variance), axis=1)
        result.append(outputs)
    return result


def process_pixel(img_array):
    """
    The same function of the keras.image.preprocess_input
    :param img_array: numpy array, the image content
    :return: the processed numpy array
    """
    img_array = img_array[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]
    return img_array


def process_input_image(image, input_dim, method=PMethod.Zoom):
    """
    Process the input image.
    :param input_dim: int, the dimension of input image
    :param image: PIL.Image object
    :param method: PMethod class
    :return: numpy array of the processed image
    """
    shape = np.array(image).shape
    if method == PMethod.Zoom:
        new_image = zoom_image(image, input_dim)
    elif method == PMethod.Reshape:
        new_image = resize_image(image, input_dim)
    else:
        raise RuntimeError("No Method Selected.")

    input_array = np.array(new_image, dtype=np.float)
    input_array = process_pixel(input_array)
    return input_array, shape


def image_show(image, input_dim, li):
    """
    Show the rectangle of the object, use for test.
    :param input_dim: int, the dimension of the input image
    :param image: PIL.Image object
    :param li: list or numpy array, must have shape (?,4) or (?,5)
    :return: None
    """
    plt.figure()
    plt.imshow(image.reshape(input_dim, input_dim, 3))
    for p in li:
        plt.gca().add_patch(
            plt.Rectangle((p[0], p[1]), p[2] - p[0],
                          p[3] - p[1], fill=False,
                          edgecolor='r', linewidth=1)
        )
    plt.show()
    plt.close()


def transform_box(box, input_dim, original_shape, method):
    """
    Change the coordinate from the original annotation.
    :param input_dim: int, the dimension of input image
    :param box: numpy array, with shape (?,5)
    :param original_shape: tuple, indicates the (x_min,y_min,x_max,y_max) of the annotation
    :param method: PMethod class
    :return: the encoded boxes, with format (x_min,y_min,x_max,y_max)
    """
    h, w, c = original_shape
    if method == PMethod.Reshape:
        box = box.astype(np.float)
        box[:, 1:4:2] = box[:, 1:4:2] / h * input_dim
        box[:, 0:4:2] = box[:, 0:4:2] / w * input_dim
        box = box.astype(np.int)
    elif method == PMethod.Zoom:
        rh = input_dim / h
        rw = input_dim / w
        ranges = min(rw, rh)
        x_min_b = np.where(box[:, 0] >= 0.5 * input_dim)
        x_min_l = np.where(box[:, 0] < 0.5 * input_dim)
        x_max_b = np.where(box[:, 2] >= 0.5 * input_dim)
        x_max_l = np.where(box[:, 2] < 0.5 * input_dim)
        y_min_b = np.where(box[:, 1] >= 0.5 * input_dim)
        y_min_l = np.where(box[:, 1] < 0.5 * input_dim)
        y_max_b = np.where(box[:, 3] >= 0.5 * input_dim)
        y_max_l = np.where(box[:, 3] < 0.5 * input_dim)
        box[x_min_b, 0] = (input_dim * 0.5 + (box[x_min_b, 0] - w * 0.5) * ranges).astype(np.int)
        box[x_min_l, 0] = (input_dim * 0.5 - (w * 0.5 - box[x_min_l, 0]) * ranges).astype(np.int)
        box[x_max_b, 2] = (input_dim * 0.5 + (box[x_max_b, 2] - w * 0.5) * ranges).astype(np.int)
        box[x_max_l, 2] = (input_dim * 0.5 - (w * 0.5 - box[x_max_l, 2]) * ranges).astype(np.int)
        box[y_min_b, 1] = (input_dim * 0.5 + (box[y_min_b, 1] - h * 0.5) * ranges).astype(np.int)
        box[y_min_l, 1] = (input_dim * 0.5 - (h * 0.5 - box[y_min_l, 1]) * ranges).astype(np.int)
        box[y_max_b, 3] = (input_dim * 0.5 + (box[y_max_b, 3] - h * 0.5) * ranges).astype(np.int)
        box[y_max_l, 3] = (input_dim * 0.5 - (h * 0.5 - box[y_max_l, 3]) * ranges).astype(np.int)
    else:
        raise RuntimeError("No Method Selected.")
    return box


def encode_label(boxes, prior_box_list, model_name):
    model = model_name.value
    prior_box = np.concatenate(prior_box_list, axis=0)
    label = np.zeros(shape=(prior_box.shape[0], 4 + len(Config.class_names) + 1 + 8), dtype=np.float32)
    label[:, 4] = 1.0
    loc = boxes[:, :4]
    pos = loc.astype(np.float) / model.input_dim
    cls = boxes[:, 4]
    for i in range(0, len(boxes)):
        index = calculate_max_iou(pos[i], prior_box)
        priors = prior_box[index]
        label[index, (cls[i] + 5)] = 1.0
        label[index, 4] = 0.0
        box_center = 0.5 * (pos[i, :2] + pos[i, 2:])
        box_wh = pos[i, 2:] - pos[i, :2]
        assigned_priors_center = 0.5 * (priors[:2] +
                                        priors[2:4])
        assigned_priors_wh = (priors[2:4] -
                              priors[:2])
        pos[i, :2] = box_center - assigned_priors_center
        pos[i, :2] /= assigned_priors_wh
        pos[i, :2] /= priors[-4:-2]

        pos[i, 2:4] = np.log(box_wh / assigned_priors_wh)
        pos[i, 2:4] /= priors[-2:]
        label[index, 0:4] = pos[i, 0:4]
        label[index, -8] = 1
    return label


def data_generator(annotation_path, prior_box_list, model_name, batch_size=4, method=PMethod.Reshape,
                   use_generator=False):
    """
    The generator for training the RPN model.
    :param model_name: ModelConfig object
    :param batch_size: int, the size of a single batch
    :param prior_box_list: list, the prior box
    :param annotation_path: str, the path to annotation file(if not using generator)
    :param method: PMethod class, the method to process the input image
    :param use_generator: bool, if use generator, the annotation file will not be used
    :return: single training data
    """
    model = model_name.value
    if use_generator:
        annotation_lines = []
        img_list = get_image_number_list()
    else:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation_lines = f.readlines()
        np.random.shuffle(annotation_lines)
        img_list = []
    X = []
    Y = []
    count = 0
    while True:
        if use_generator:
            image, box = generate_single_image(img_list)
            x, o_s = process_input_image(image, model.input_dim, method)
            u = transform_box(box, model.input_dim, o_s, method)
            y = encode_label(u, prior_box_list, model_name)
            X.append(x)
            Y.append(y)
            count += 1
            if count == batch_size:
                count = 0
                yield np.array(X), np.array(Y)
                X = []
                Y = []
        else:
            for term in annotation_lines:
                line = term.split()
                img_path = line[0]
                img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
                image = Image.open(img_path)
                x, o_s = process_input_image(image, model.input_dim, method)
                u = transform_box(img_box, model.input_dim, o_s, method)
                y = encode_label(u, prior_box_list, model_name)
                X.append(x)
                Y.append(y)
                count += 1
                if count == batch_size:
                    count = 0
                    yield np.array(X), np.array(Y)
                    X = []
                    Y = []


def get_weight_file(name):
    """
    Return the weight file in the logs.
    :param name: str, the name of weight file
    :return: str, the path to selected weight file
    """
    file_list = []
    for root, dirs, files in os.walk(Config.checkpoint_dir):
        for d in dirs:
            dir_path = os.path.join(root, d)
            for _r, _d, f in os.walk(dir_path):
                for dx in _d:
                    _dir = os.path.join(_r, dx)
                    for _root, _x, weight_files in os.walk(_dir):
                        for weight_file in weight_files:
                            if weight_file == name:
                                file_list.append(os.path.join(dir_path, weight_file))
                            elif name + ".h5" == weight_file:
                                file_list.append(os.path.join(dir_path, weight_file))
    for root, dirs, files in os.walk(Config.weight_dir):
        for weight_file in files:
            if weight_file == name:
                file_list.append(os.path.join(root, weight_file))
            elif name + ".h5" == weight_file:
                file_list.append(os.path.join(root, weight_file))
    if len(file_list) == 0:
        raise RuntimeError("No weight suitable.")
    else:
        if len(file_list) == 1:
            print("Use weight in {}".format(file_list[0]))
            return file_list[0]
        else:
            recent_file = None
            recent_time = 0
            for f in file_list:
                time = os.path.getctime(f)
                if time > recent_time:
                    recent_file = f
                    recent_time = time
            if recent_file is not None:
                print("Use weight in {}".format(recent_file))
                return recent_file
            else:
                raise RuntimeError("No weight suitable.")
