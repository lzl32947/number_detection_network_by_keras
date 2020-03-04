from parameter.parameters import HyperParameter
from util.img_process_util import process_single_input
import matplotlib.pyplot  as plt
import tensorflow as tf
import keras.backend as K

from networks.ssd_linked import get_SSD_model
from util.decode_util import *
from util.drawing_util import draw_image
from util.img_process_util import process_single_input

if __name__ == '__main__':
    img_path = "./img/number2.jpeg"
    photo, x_offset_ratio, y_offset_ratio, img = process_single_input(img_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    model = get_SSD_model()
    plt.figure()
    plt.imshow(photo.reshape((HyperParameter.min_dim, HyperParameter.min_dim, 3)))
    plt.show()
    plt.close()
    model.load_weights("./checkpoints/best.h5", by_name=True)
    result_list = model.predict(photo)
    result_decode = decode_predict(result_list)
    box_result = adjust_prediction_region(result_decode)
    for image_batch in box_result:
        for k in image_batch:
            if np.argmax(k['conf']) != 0:
                print(k)
            # nms_list = process_nms(sess, image_batch)
            # draw_image(img, nms_list, offset_x=x_offset_ratio, offset_y=y_offset_ratio)
    sess.close()
