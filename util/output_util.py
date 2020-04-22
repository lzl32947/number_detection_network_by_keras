import tensorflow as tf
import numpy as np

from config.Configs import Config
import keras.backend as K


def decode_result(result_array, model_name):
    model = model_name.value
    assert len(result_array.shape) == 3

    boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    scores = tf.placeholder(dtype='float32', shape=(None,))

    return_list = []
    for batch in result_array:
        # softmax filter
        not_bg = batch[np.where(batch[:, 4] < model.softmax_threshold)]
        # sort by class
        for index in range(0, len(Config.class_names)):
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
                                                                              model.top_k,
                                                                              iou_threshold=model.nms_threshold),
                                                 feed_dict={
                                                     boxes: decode_box,
                                                     scores: conf
                                                 })
                for k in zip(decode_box[nms_result], conf[nms_result]):
                    return_list.append([k[0], k[1], index])
    return return_list
