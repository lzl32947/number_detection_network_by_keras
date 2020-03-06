import os

import tensorflow as tf
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image
from networks.ssd_linked import get_SSD_model
from util.decode_util import *
from util.drawing_util import draw_image
from util.img_process_util import process_single_input

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    model = get_SSD_model()
    model.load_weights("./checkpoints/ep016-loss0.005-val_loss0.029_more.h5", by_name=True)
    for root, dirs, files in os.walk(r"G:\data_stored\test_line"):
        for file in files:
            img_path = os.path.join(root, file)
            # img_path = "./img/street2.jpg.jpg"
            image = Image.open(img_path)
            image = image.resize((300, 300), Image.ANTIALIAS)
            photo, x_offset_ratio, y_offset_ratio, img = process_single_input(image)
            # k = img.resize((300, 300), Image.ANTIALIAS)
            # photo = np.reshape(np.array(k), (1, HyperParameter.min_dim, HyperParameter.min_dim, 3))
            # plt.figure()
            # plt.imshow(photo)
            # plt.show()
            # plt.close()
            photo = photo.reshape((1, HyperParameter.min_dim, HyperParameter.min_dim, 3))
            result_list = model.predict(photo)
            result_decode = decode_predict(result_list)
            box_result = adjust_prediction_region(result_decode)
            for image_batch in box_result:
                nms_list = process_nms(sess, image_batch)
                print(nms_list)
                draw_image(img, nms_list, offset_x=x_offset_ratio, offset_y=y_offset_ratio)
    sess.close()
