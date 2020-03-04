import pickle
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from networks.ssd_linked import get_SSD_model
from parameter.parameters import HyperParameter
from archived.input_captcha_util import raw_data_generator


def calculate_max_iou(priors_array, x_min, y_min, x_max, y_max):
    index = 0
    iou_max = 0
    for underline in range(len(priors_array)):
        position = priors_array[underline]
        p_x_min = int(position[0] * HyperParameter.min_dim)
        p_y_min = int(position[1] * HyperParameter.min_dim)
        p_x_max = int(position[2] * HyperParameter.min_dim)
        p_y_max = int(position[3] * HyperParameter.min_dim)

        inter_width = min(x_max, p_x_max) - max(x_min, p_x_min)
        inter_height = min(y_max, p_y_max) - max(y_min, p_y_min)
        inter_area = max(0, inter_height) * max(0, inter_width)
        full_area = (y_max - y_min) * (x_max - x_min) + (p_y_max - p_y_min) * (p_x_max - p_x_min)
        iou = inter_area / (full_area - inter_area)
        if iou > iou_max:
            iou_max = iou
            index = underline
    return index


def generate_single_data(dir_path, priors):
    # priors with shape(8732,8), which is the shape of all prior box in the last
    for image, position_dict in raw_data_generator(dir_path):
        image = image.reshape((HyperParameter.min_dim, HyperParameter.min_dim, 3))
        result_array = np.zeros(shape=(priors.shape[0], HyperParameter.class_num + 4), dtype="float32")
        # all background
        result_array[:, 4] = 1.0

        for digit in position_dict:
            digit_class = 1 + digit['name']
            digit_x_min = digit['xmin']
            digit_x_max = digit['xmax']
            digit_y_min = digit['ymin']
            digit_y_max = digit['ymax']
            index = calculate_max_iou(priors, digit_x_min, digit_y_min, digit_x_max, digit_y_max)

            digit_x_min /= HyperParameter.min_dim
            digit_x_max /= HyperParameter.min_dim
            digit_y_min /= HyperParameter.min_dim
            digit_y_max /= HyperParameter.min_dim
            p_x_min = priors[index, 0]
            p_y_min = priors[index, 1]
            p_x_max = priors[index, 2]
            p_y_max = priors[index, 3]
            real_box_center_x = (digit_x_min + digit_x_max) / 2
            real_box_center_y = (digit_y_min + digit_y_max) / 2
            real_box_width = digit_x_max - digit_x_min
            real_box_height = digit_y_max - digit_y_min
            prior_box_center_x = (p_x_min + p_x_max) / 2
            prior_box_center_y = (p_y_min + p_y_max) / 2
            prior_box_width = p_x_max - p_x_min
            prior_box_height = p_y_max - p_y_min
            adjust_x = real_box_center_x - prior_box_center_x
            adjust_x /= prior_box_width
            adjust_x /= HyperParameter.variances[0]
            adjust_y = real_box_center_y - prior_box_center_y
            adjust_y /= prior_box_height
            adjust_y /= HyperParameter.variances[1]
            adjust_width = real_box_width / prior_box_width
            adjust_width = np.log(adjust_width)
            adjust_width /= HyperParameter.variances[2]
            adjust_height = real_box_height / prior_box_height
            adjust_height = np.log(adjust_height)
            adjust_height /= HyperParameter.variances[3]
            # 组装
            result_array[index, 0] = adjust_x
            result_array[index, 1] = adjust_y
            result_array[index, 2] = adjust_width
            result_array[index, 3] = adjust_height
            result_array[index, 4 + digit_class] = 1.0
            result_array[index, 4] = 0.0
        result = np.concatenate((result_array, priors), axis=1)
        yield image, result


def data_generator(dir_path, priors, batch_size=4):
    X = []
    Y = []
    count = 0
    while True:
        for img, result in generate_single_data(dir_path, priors):
            X.append(img)
            Y.append(result)
            count += 1
            if count == batch_size:
                yield np.array(X), np.array(Y)
                count = 0
                X = []
                Y = []


def loss_function(y_true, y_pred):
    def _l1_smooth_loss(y_t, y_p):
        abs_loss = tf.abs(y_t - y_p)
        sq_loss = 0.5 * (y_t - y_p) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(y_t, y_p):
        y_p = tf.maximum(tf.minimum(y_p, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_t * tf.log(y_p),
                                      axis=-1)
        return softmax_loss

    batch_size = tf.shape(y_true)[0]
    num_boxes = tf.to_float(tf.shape(y_true)[1])

    # 计算所有的loss
    # 分类的loss
    # batch_size,8732,21 -> batch_size,8732
    conf_loss = _softmax_loss(y_true[:, :, 4:-8],
                              y_pred[:, :, 4:-8])
    # 框的位置的loss
    # batch_size,8732,4 -> batch_size,8732
    loc_loss = _l1_smooth_loss(y_true[:, :, :4],
                               y_pred[:, :, :4])

    # 获取所有的正标签的loss
    # 每一张图的pos的个数
    num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
    # 每一张图的pos_loc_loss
    pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],
                                 axis=1)
    # 每一张图的pos_conf_loss
    pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],
                                  axis=1)

    # 获取一定的负样本
    num_neg = tf.minimum(HyperParameter.neg_pos_ratio * num_pos,
                         num_boxes - num_pos)

    # 找到了哪些值是大于0的
    pos_num_neg_mask = tf.greater(num_neg, 0)
    # 获得一个1.0
    has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
    num_neg = tf.concat(axis=0, values=[num_neg,
                                        [(1 - has_min) * HyperParameter.negatives_for_hard]])
    # 求平均每个图片要取多少个负样本
    num_neg_batch = tf.reduce_mean(tf.boolean_mask(num_neg,
                                                   tf.greater(num_neg, 0)))
    num_neg_batch = tf.to_int32(num_neg_batch)

    # conf的起始
    confs_start = 4 + 1
    # conf的结束
    confs_end = confs_start + HyperParameter.class_num - 1

    # 找到实际上在该位置不应该有预测结果的框，求他们最大的置信度。
    max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                              axis=2)

    # 取top_k个置信度，作为负样本
    _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]),
                             k=num_neg_batch)

    # 找到其在1维上的索引
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                    tf.reshape(indices, [-1]))

    neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                              full_indices)
    neg_conf_loss = tf.reshape(neg_conf_loss,
                               [batch_size, num_neg_batch])
    neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

    # 求loss总和
    total_loss = pos_conf_loss + neg_conf_loss
    total_loss /= (num_pos + tf.to_float(num_neg_batch))
    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                       tf.ones_like(num_pos))
    total_loss += (HyperParameter.alpha * pos_loc_loss) / num_pos
    return total_loss


if __name__ == '__main__':
    priors = pickle.load(open('model_data/prior_boxes_ssd300.pkl', 'rb'))
    log_dir = "logs/"
    weight_dir = "checkpoints/"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    K.set_session(sess)

    model = get_SSD_model()
    model.load_weights("weight/ssd_weights.h5", by_name=True, skip_mismatch=True)
    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(weight_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    BATCH_SIZE = 3

    if True:
        model.compile(optimizer=Adam(lr=1e-5), loss=loss_function)
        model.fit_generator(data_generator("./data/train", priors, batch_size=BATCH_SIZE),
                            steps_per_epoch=150 // BATCH_SIZE,
                            validation_data=data_generator("./data/test", priors, batch_size=BATCH_SIZE),
                            validation_steps=15 // BATCH_SIZE,
                            epochs=10,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
