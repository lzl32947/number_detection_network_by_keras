import os
import keras
from Configs import Config, PMethod
from models import choose_model
from util.data_util import process_data, decode_result
from util.image_util import draw_image

if __name__ == '__main__':
    process_method = PMethod.Zoom
    model = choose_model("vgg16", 20)
    model.load_weights(os.path.join(Config.weight_dir, "ssd_weights_20.h5"), by_name=True, skip_mismatch=True)
    image_path = os.path.join(Config.test_data_dir, "street2.jpg")
    x, shape = process_data(image_path,process_method)
    result = model.predict(x)
    result_list = decode_result(result)
    draw_image(image_path,result_list,process_method)