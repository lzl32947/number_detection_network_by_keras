from PIL import Image
import numpy as np
import os
import tensorflow as tf
import keras.backend as K
from Configs import Config, PMethod
from util.image_util import zoom_image, resize_image

import matplotlib.pyplot as plt


def calculate_max_iou(pos):
    assert Config.priors is not None
    pri = Config.priors * Config.input_dim
    pri = pri.astype(np.int)
    _pos = np.expand_dims(pos, axis=0).repeat(len(pri), axis=0)
    inter_width = np.minimum(_pos[:, 2], pri[:, 2]) - np.maximum(_pos[:, 0], pri[:, 0])
    inter_height = np.minimum(_pos[:, 3], pri[:, 3]) - np.maximum(_pos[:, 1], pri[:, 1])
    inter_area = np.maximum(0, inter_height) * np.maximum(0, inter_width)
    full_area = (_pos[:, 3] - _pos[:, 1]) * (_pos[:, 2] - _pos[:, 0]) + (pri[:, 3] - pri[:, 1]) * (
            pri[:, 2] - pri[:, 0])
    iou = inter_area / (full_area - inter_area)
    return np.argmax(iou)


def encode_box(box, original_shape, method):
    h, w, c = original_shape
    if method == PMethod.Reshape:
        box = box.astype(np.float)
        box[:, 1:4:2] = box[:, 1:4:2] / h * Config.input_dim
        box[:, 0:4:2] = box[:, 0:4:2] / w * Config.input_dim
        box = box.astype(np.int)
    elif method == PMethod.Zoom:
        rh = Config.input_dim / h
        rw = Config.input_dim / w
        ranges = min(rw, rh)
        x_min_b = np.where(box[:, 0] >= 0.5 * Config.input_dim)
        x_min_l = np.where(box[:, 0] < 0.5 * Config.input_dim)
        x_max_b = np.where(box[:, 2] >= 0.5 * Config.input_dim)
        x_max_l = np.where(box[:, 2] < 0.5 * Config.input_dim)
        y_min_b = np.where(box[:, 1] >= 0.5 * Config.input_dim)
        y_min_l = np.where(box[:, 1] < 0.5 * Config.input_dim)
        y_max_b = np.where(box[:, 3] >= 0.5 * Config.input_dim)
        y_max_l = np.where(box[:, 3] < 0.5 * Config.input_dim)
        box[x_min_b, 0] = (Config.input_dim * 0.5 + (box[x_min_b, 0] - w * 0.5) * ranges).astype(np.int)
        box[x_min_l, 0] = (Config.input_dim * 0.5 - (w * 0.5 - box[x_min_l, 0]) * ranges).astype(np.int)
        box[x_max_b, 2] = (Config.input_dim * 0.5 + (box[x_max_b, 2] - w * 0.5) * ranges).astype(np.int)
        box[x_max_l, 2] = (Config.input_dim * 0.5 - (w * 0.5 - box[x_max_l, 2]) * ranges).astype(np.int)
        box[y_min_b, 1] = (Config.input_dim * 0.5 + (box[y_min_b, 1] - h * 0.5) * ranges).astype(np.int)
        box[y_min_l, 1] = (Config.input_dim * 0.5 - (h * 0.5 - box[y_min_l, 1]) * ranges).astype(np.int)
        box[y_max_b, 3] = (Config.input_dim * 0.5 + (box[y_max_b, 3] - h * 0.5) * ranges).astype(np.int)
        box[y_max_l, 3] = (Config.input_dim * 0.5 - (h * 0.5 - box[y_max_l, 3]) * ranges).astype(np.int)
    else:
        raise RuntimeError("No Method Selected.")
    return box


def encode_label(box):
    assert Config.priors is not None
    label = np.zeros(shape=(Config.priors.shape[0], 4 + Config.class_num + 8), dtype=np.float32)
    label[:, 4] = 1.0
    loc = box[:, :4]
    pos = loc.astype(np.float) / Config.input_dim
    cls = box[:, 4]
    for i in range(0, len(box)):
        index = calculate_max_iou(loc[i])
        priors = Config.priors[index]
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


def data_generator(annotation_path, batch_size=4, method=PMethod.Zoom):
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation_lines = f.readlines()
    np.random.shuffle(annotation_lines)
    X = []
    Y = []
    count = 0
    while True:
        for term in annotation_lines:
            line = term.split()
            img_path = line[0]
            img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
            x, o_s = process_input_image(img_path, method)
            u = encode_box(img_box, o_s, method)
            y = encode_label(u)
            X.append(x)
            Y.append(y)
            count += 1
            if count == batch_size:
                count = 0
                yield np.array(X), np.array(Y)
                X = []
                Y = []


def decode_result(result_array):
    boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    scores = tf.placeholder(dtype='float32', shape=(None,))
    assert len(result_array.shape) == 3
    return_list = []
    for batch in result_array:
        # softmax filter
        not_bg = batch[np.where(batch[:, 4] < Config.softmax_threshold)]
        # sort by class
        for index in range(0, Config.class_num - 1):
            # find the result
            item = not_bg[np.where(np.argmax(not_bg[:, 5:-8], axis=1) == index)]
            if len(item) != 0:
                priorbox = item[:, -8:-4]
                pred_loc = item[:, :4]
                variance = item[:, -4:]
                # get the width and height of prior box
                prior_width = priorbox[:, 2] - priorbox[:, 0]
                prior_height = priorbox[:, 3] - priorbox[:, 1]

                # get the center point of prior box
                prior_center_x = 0.5 * (priorbox[:, 2] + priorbox[:, 0])
                prior_center_y = 0.5 * (priorbox[:, 3] + priorbox[:, 1])

                # get the offset of the prior box and the real box
                decode_box_center_x = pred_loc[:, 0] * prior_width * variance[:, 0]
                decode_box_center_x += prior_center_x
                decode_box_center_y = pred_loc[:, 1] * prior_height * variance[:, 1]
                decode_box_center_y += prior_center_y

                # get the width and height of the real box
                decode_box_width = np.exp(pred_loc[:, 2] * variance[:, 2])
                decode_box_width *= prior_width
                decode_box_height = np.exp(pred_loc[:, 3] / 4 * variance[:, 3])
                decode_box_height *= prior_height

                # get the top-left and bottom-right of the real box
                decode_box_xmin = decode_box_center_x - 0.5 * decode_box_width
                decode_box_ymin = decode_box_center_y - 0.5 * decode_box_height
                decode_box_xmax = decode_box_center_x + 0.5 * decode_box_width
                decode_box_ymax = decode_box_center_y + 0.5 * decode_box_height

                # concatenate the result
                decode_box = np.concatenate((decode_box_xmin[:, None],
                                             decode_box_ymin[:, None],
                                             decode_box_xmax[:, None],
                                             decode_box_ymax[:, None]), axis=-1)
                # clip the result by 0 and 1
                decode_box = np.minimum(np.maximum(decode_box, 0.0), 1.0)
                conf = item[:, 5 + index]

                nms_result = K.get_session().run(tf.image.non_max_suppression(boxes, scores,
                                                                              Config.top_k,
                                                                              iou_threshold=Config.nms_threshold),
                                                 feed_dict={
                                                     boxes: decode_box,
                                                     scores: conf
                                                 })
                for k in zip(decode_box[nms_result], conf[nms_result]):
                    return_list.append([k[0], k[1], index])
    return return_list


def process_pixel(img_array):
    # 该函数旨在简化keras中preprocess_input函数的工作过程
    # 和preprocess_input函数返回相同的值
    # 'RGB'->'BGR'
    img_array = img_array[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]
    return img_array


def process_input_image(image_path, method=PMethod.Zoom):
    image = Image.open(image_path)
    shape = np.array(image).shape
    if method == PMethod.Zoom:
        new_image = zoom_image(image)
    elif method == PMethod.Reshape:
        new_image = resize_image(image)
    else:
        raise RuntimeError("No Method Selected.")

    input_array = np.array(new_image, dtype=np.float)
    input_array = process_pixel(input_array)
    return input_array, shape


def test(image, li):
    plt.figure()
    plt.imshow(image.reshape(Config.input_dim, Config.input_dim, 3))
    for p in li:
        plt.gca().add_patch(
            plt.Rectangle((p[0], p[1]), p[2] - p[0],
                          p[3] - p[1], fill=False,
                          edgecolor='r', linewidth=1)
        )
    plt.show()
    plt.close()
