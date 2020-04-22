from config.Configs import PMethod, ModelConfig
from train import get_model, init_session
from util.image_generator import generate_single_image, get_image_number_list
from util.image_util import draw_image
from util.input_util import process_input_image
import numpy as np
from PIL import Image

from util.output_util import decode_result

if __name__ == '__main__':
    init_session()
    model = ModelConfig.VGG16
    process_method = PMethod.Reshape
    predict_model = get_model(weight_file=[], load_by_name=[], model_name=model)
    img_list = get_image_number_list()
    while True:
        image, box = generate_single_image(img_list)
        x, shape = process_input_image(image, model.value.input_dim, process_method)
        x = np.expand_dims(x, axis=0)
        result = model.predict(x)
        result_list = decode_result(result, model_name=model)
        draw_image(image, result_list, process_method, model.value.input_dim)
