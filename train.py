import os
from datetime import datetime
import keras
import tensorflow as tf
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model

from config.Configs import Config, PMethod, ModelConfig
from model.resnet50_base import ResNet50_SSD
from model.vgg16_base import VGG16_SSD
from model.resnet101_base import ResNet101_SSD
from model.mobilenetv2_base import MobileNetV2_SSD
from util.input_util import data_generator, get_prior_box_list, get_weight_file


def loss_function(y_t, y_p):
    def _l1_smooth_loss(y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(y_true, y_pred):
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return softmax_loss

    batch_size = tf.shape(y_t)[0]
    num_boxes = tf.to_float(tf.shape(y_t)[1])
    # 计算所有的loss
    # 分类的loss
    # batch_size,8732,21 -> batch_size,8732
    conf_loss = _softmax_loss(y_t[:, :, 4:-8], y_p[:, :, 4:-8])
    # 框的位置的loss
    # batch_size,8732,4 -> batch_size,8732
    loc_loss = _l1_smooth_loss(y_t[:, :, :4], y_p[:, :, :4])
    # 获取所有的正标签的loss
    # 每一张图的pos的个数
    num_pos = tf.reduce_sum(y_t[:, :, -8], axis=-1)
    # 每一张图的pos_loc_loss
    pos_loc_loss = tf.reduce_sum(loc_loss * y_t[:, :, -8], axis=1)
    # 每一张图的pos_conf_loss
    pos_conf_loss = tf.reduce_sum(conf_loss * y_t[:, :, -8], axis=1)
    # 获取一定的负样本
    num_neg = tf.minimum(Config.neg_pos_ratio * num_pos, num_boxes - num_pos)
    # 找到了哪些值是大于0的
    pos_num_neg_mask = tf.greater(num_neg, 0)
    # 获得一个1.0
    has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
    num_neg = tf.concat(axis=0, values=[num_neg, [(1 - has_min) * Config.negatives_for_hard]])
    # 求平均每个图片要取多少个负样本
    num_neg_batch = tf.reduce_mean(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
    num_neg_batch = tf.to_int32(num_neg_batch)
    # conf的起始
    confs_start = 4 + Config.bg_class + 1
    # conf的结束
    confs_end = confs_start + len(Config.class_names)
    # 找到实际上在该位置不应该有预测结果的框，求他们最大的置信度。
    max_confs = tf.reduce_max(y_p[:, :, confs_start:confs_end],
                              axis=2)
    # 取top_k个置信度，作为负样本
    _, indices = tf.nn.top_k(max_confs * (1 - y_t[:, :, -8]), k=num_neg_batch)
    # 找到其在1维上的索引
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                    tf.reshape(indices, [-1]))
    neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
    neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
    neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)
    # 求loss总和
    total_loss = pos_conf_loss + neg_conf_loss
    total_loss /= (num_pos + tf.to_float(num_neg_batch))
    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
    total_loss += (Config.alpha * pos_loc_loss) / num_pos
    return total_loss


def init_session():
    """
    Init the tensorflow session.
    :return: None
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    K.set_session(sess)


def get_model(weight_file, load_by_name, model_name, show_image=False):
    prior_box_list = get_prior_box_list(model_name)
    if model_name == ModelConfig.VGG16:
        model = VGG16_SSD(prior_box_list)
    elif model_name == ModelConfig.ResNet50:
        model = ResNet50_SSD(prior_box_list)
    elif model_name == ModelConfig.ResNet101:
        model = ResNet101_SSD(prior_box_list)
    elif model_name == ModelConfig.MobileNetV2:
        model = MobileNetV2_SSD(prior_box_list)
    else:
        raise RuntimeError("No model selected.")
    if weight_file:
        for i, j in zip(weight_file, load_by_name):
            model.load_weights(i, by_name=j, skip_mismatch=True)
    if show_image:
        plot_model(model, to_file=os.path.join(Config.model_output_dir, "{}.png".format(model_name)),
                   show_shapes=True,
                   show_layer_names=True)
    return model


def train(weight_file_list, load_weight_by_name, model_name, use_generator=False, show_image=False, show_summary=False,
          batch_size=4):
    init_session()
    model = get_model(weight_file=weight_file_list, show_image=show_image, model_name=model_name,
                      load_by_name=load_weight_by_name)

    if show_summary:
        model.summary(line_length=100)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=1e-4))
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(Config.checkpoint_dir, str(model_name))
    log_dir = os.path.join(Config.tensorboard_log_dir, str(model_name))
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, time)
    log_dir = os.path.join(log_dir, time)
    os.mkdir(checkpoint_dir)
    os.mkdir(log_dir)
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir,
                     'SSD_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-' + str(model_name) + '.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    prior_box_list = get_prior_box_list(model_name)
    model.fit_generator(
        data_generator(Config.train_annotation_path, method=PMethod.Reshape,
                       prior_box_list=prior_box_list, model_name=model_name,
                       use_generator=use_generator, batch_size=batch_size),
        steps_per_epoch=1000,
        epochs=500,
        validation_data=data_generator(Config.valid_annotation_path, method=PMethod.Reshape,
                                       prior_box_list=prior_box_list, model_name=model_name,
                                       use_generator=use_generator, batch_size=batch_size),
        validation_steps=100,
        initial_epoch=0,
        callbacks=[logging, checkpoint],
        verbose=1)


if __name__ == '__main__':
    train(weight_file_list=[get_weight_file("ssd_weights.h5"),
                            get_weight_file("resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5")],
          load_weight_by_name=[True, True],
          model_name=ModelConfig.ResNet101,
          show_image=True,
          batch_size=2,
          use_generator=True,
          show_summary=True)
