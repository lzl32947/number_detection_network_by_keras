import numpy as np
import tensorflow as tf

from parameter.parameters import HyperParameter


def decode_predict(predictions):
    result = []
    for i in range(0, predictions.shape[0]):
        single_prediction = []
        for j in range(0, predictions.shape[1]):
            net = {}
            net['loc'] = predictions[i, j, :4]
            net['variances'] = predictions[i, j, -4:]
            net['priorbox'] = predictions[i, j, -8:-4]
            net['conf'] = predictions[i, j, 4:-8]
            net['id'] = j
            single_prediction.append(net)
        result.append(single_prediction)
    return result


def adjust_prediction_region(result_list):
    for img_prediction in result_list:
        for single_prediction in img_prediction:
            # get the prior box data
            prior_width = single_prediction['priorbox'][2] - single_prediction['priorbox'][0]
            prior_height = single_prediction['priorbox'][3] - single_prediction['priorbox'][1]
            prior_center_x = (single_prediction['priorbox'][2] + single_prediction['priorbox'][0]) * 0.5
            prior_center_y = (single_prediction['priorbox'][3] + single_prediction['priorbox'][1]) * 0.5
            # get the adjust data of box
            # get box center
            box_adjust_x = single_prediction['loc'][0] * prior_width * single_prediction['variances'][0]
            box_center_x = box_adjust_x + prior_center_x
            box_adjust_y = single_prediction['loc'][1] * prior_height * single_prediction['variances'][1]
            box_center_y = box_adjust_y + prior_center_y
            # get box width and height
            box_adjust_width = np.exp(single_prediction['loc'][2] * single_prediction['variances'][2])
            box_adjust_height = np.exp(single_prediction['loc'][3] * single_prediction['variances'][3])
            box_width = box_adjust_width * prior_width
            box_height = box_adjust_height * prior_height
            # get box left-top and right-bottom
            box_x_min = box_center_x - box_width * 0.5
            box_y_min = box_center_y - box_height * 0.5
            box_x_max = box_center_x + box_width * 0.5
            box_y_max = box_center_y + box_height * 0.5
            # return
            new_loc = np.clip([box_x_min, box_y_min, box_x_max, box_y_max], 0.0, 1.0)
            single_prediction['new_loc'] = new_loc
    return result_list


def process_nms(session, new_list):
    return_list = []
    # 按类别分类，默认0是背景类别
    for class_index in range(0, HyperParameter.class_num):
        if class_index == HyperParameter.bg_class:
            continue
        # 得到每个类别的置信度高于threshold的图相框和其置信度大小
        box_list = []
        conf_list = []
        for single_prediction in new_list:
            if np.argmax(single_prediction['conf']) == class_index and \
                    single_prediction['conf'][np.argmax(single_prediction['conf'])] > HyperParameter.threshold:
                box_list.append(single_prediction['new_loc'])
                conf_list.append(single_prediction['conf'][np.argmax(single_prediction['conf'])])
        if len(box_list) > 0:
            # 构建tf网络并运行非极大值抑制
            boxes = tf.placeholder(dtype='float32', shape=(None, 4))
            scores = tf.placeholder(dtype='float32', shape=(None,))
            nms_result = session.run(tf.image.non_max_suppression(boxes, scores,
                                                                  HyperParameter.top_k,
                                                                  iou_threshold=HyperParameter.threshold),
                                     feed_dict={
                                         boxes: box_list,
                                         scores: conf_list
                                     })
            result_dict = {}
            result_dict['class'] = class_index
            box_loc_list = []
            label_list = []
            for i in nms_result:
                box_loc_list.append(box_list[i])
                label_list.append(conf_list[i])
            result_dict['box'] = box_loc_list
            result_dict['conf'] = label_list
            return_list.append(result_dict)
    return return_list
