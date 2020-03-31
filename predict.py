import os

from config.Configs import PMethod, Config
from models import SSD_Model
from util.data_util import decode_result, process_input_image
from util.image_util import draw_image
import numpy as np

if __name__ == '__main__':
    process_method = PMethod.Reshape
    model = SSD_Model("vgg16", 10, [os.path.join(Config.checkpoint_dir, "20200331_134926\\ep051-loss0.787-val_loss0.666.h5"), ])
    for root, dirs, files in os.walk(r"G:\data_stored\nature_train"):
        for img in files:
            image_path = os.path.join(root, img)
            x, shape = process_input_image(image_path, process_method)
            x = np.expand_dims(x, axis=0)
            result = model.predict(x)
            result_list = decode_result(result)
            draw_image(image_path, result_list, process_method)
