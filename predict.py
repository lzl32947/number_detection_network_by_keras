from datetime import datetime

from config.Configs import PMethod, ModelConfig, Config
from train import get_model, init_session
from util.image_generator import generate_single_image, get_image_number_list
from util.image_util import draw_image
from util.input_util import process_input_image, get_weight_file
import numpy as np
from PIL import Image

from util.output_util import decode_result
import os


def prediction_for_recording(record_name, weight_file, model_name,
                             use_generator=True, generator_count=0, use_annotation=True, annotation_lines=[],
                             save_image=False, method=PMethod.Reshape):
    init_session()
    predict_model = get_model(
        weight_file=weight_file,
        load_by_name=[True for i in weight_file], model_name=model_name)
    image_list = get_image_number_list()

    if not os.path.exists(Config.detection_result_dir):
        os.mkdir(Config.detection_result_dir)
    record_dir = os.path.join(Config.detection_result_dir, record_name)
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    time_dir = os.path.join(record_dir, time)
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)
    gt_dir = os.path.join(time_dir, "ground-truth")
    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)
    img_dir = os.path.join(time_dir, "images-optional")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    result_dir = os.path.join(time_dir, "detection-results")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    def predict_on_single_batch(image, predict_model, model_name, method=PMethod.Reshape):
        model = model_name.value
        x, shape = process_input_image(image, model.input_dim, method)
        x = np.expand_dims(x, axis=0)
        result = predict_model.predict(x)
        result_list = decode_result(result, model_name=model_name)
        return result_list

    def decode_result_list(image, results, model_name, method=PMethod.Reshape):
        input_dim = model_name.value.input_dim
        shape = np.array(image).shape
        width_zoom_ratio = input_dim / shape[1]
        height_zoom_ratio = input_dim / shape[0]
        zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
        new_width = int(zoom_ratio * shape[1])
        new_height = int(zoom_ratio * shape[0])

        width_offset = (input_dim - new_width) // 2
        height_offset = (input_dim - new_height) // 2
        width_offset /= input_dim
        height_offset /= input_dim
        for box, conf, index in results:
            if method == PMethod.Reshape:
                x_min = shape[1] * box[0]
                y_min = shape[0] * box[1]
                x_max = shape[1] * box[2]
                y_max = shape[0] * box[3]
            elif method == PMethod.Zoom:
                x_min = input_dim * box[0]
                y_min = input_dim * box[1]
                x_max = input_dim * box[2]
                y_max = input_dim * box[3]
                if width_offset > 0:
                    if x_min < input_dim * 0.5:
                        x_min = input_dim * 0.5 - (
                                (input_dim * 0.5 - x_min) * shape[0] / shape[1])
                    else:
                        x_min = input_dim * 0.5 + (x_min - input_dim * 0.5) * shape[0] / \
                                shape[1]
                    if x_max < input_dim * 0.5:
                        x_max = input_dim * 0.5 - (
                                (input_dim * 0.5 - x_max) * shape[0] / shape[1])
                    else:
                        x_max = input_dim * 0.5 + (x_max - input_dim * 0.5) * shape[0] / \
                                shape[1]
                if height_offset > 0:
                    if y_min < input_dim * 0.5:
                        y_min = input_dim * 0.5 - (
                                (input_dim * 0.5 - y_min) * shape[1] / shape[0])
                    else:
                        y_min = input_dim * 0.5 + (y_min - input_dim * 0.5) * shape[1] / \
                                shape[0]
                    if y_max < input_dim * 0.5:
                        y_max = input_dim * 0.5 - (
                                (input_dim * 0.5 - y_max) * shape[1] / shape[0])
                    else:
                        y_max = input_dim * 0.5 + (y_max - input_dim * 0.5) * shape[1] / \
                                shape[0]
                x_min = x_min / input_dim * shape[1]
                x_max = x_max / input_dim * shape[1]
                y_min = y_min / input_dim * shape[0]
                y_max = y_max / input_dim * shape[0]
            else:
                raise RuntimeError("No Method Selected.")
            yield int(index), conf, int(x_min), int(y_min), int(x_max), int(y_max)

    count = 1

    if use_annotation:
        for term in annotation_lines:
            line = term.split()
            img_path = line[0]
            img_box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]], dtype=np.int)
            img = Image.open(img_path)
            result_list = predict_on_single_batch(img, predict_model, model_name, method)
            batch_name = "a{:0>4d}".format(count)
            with open(os.path.join(gt_dir, batch_name + ".txt"), "w") as f_gt:
                for line in img_box:
                    f_gt.write("{} {} {} {} {}\n".format(line[4], line[0], line[1], line[2], line[3]))
            with open(os.path.join(result_dir, batch_name + ".txt"), "w") as f_res:
                if len(result_list) > 0:
                    for i, c, xi, yi, xa, ya in decode_result_list(img, result_list, model_name, method):
                        f_res.write("{} {} {} {} {} {}\n".format(i, c, xi, yi, xa, ya))
            if save_image:
                img.save(os.path.join(img_dir, batch_name + ".png"))
            count += 1
            if count % 10 == 0:
                print("finish {} image for annotation".format(count))
    if use_generator:
        for count in range(1, generator_count + 1):
            raw_image, raw_list = generate_single_image(image_list)
            result_list = predict_on_single_batch(raw_image, predict_model, model_name, method)
            batch_name = "g{:0>4d}".format(count)
            with open(os.path.join(gt_dir, batch_name + ".txt"), "w") as f_gt:
                for line in raw_list:
                    f_gt.write("{} {} {} {} {}\n".format(line[4], line[0], line[1], line[2], line[3]))
            with open(os.path.join(result_dir, batch_name + ".txt"), "w") as f_res:
                if len(result_list) > 0:
                    for i, c, xi, yi, xa, ya in decode_result_list(raw_image, result_list, model_name, method):
                        f_res.write("{} {} {} {} {} {}\n".format(i, c, xi, yi, xa, ya))
            if save_image:
                raw_image.save(os.path.join(img_dir, batch_name + ".png"))
            if count % 10 == 0:
                print("finish {} image for generator".format(count))


def predict_for_image(weight_file_list, load_by_name_list, model_name, method):
    model = model_name.value
    init_session()
    process_method = method
    predict_model = get_model(
        weight_file=weight_file_list,
        load_by_name=load_by_name_list, model_name=model_name)
    img_list = get_image_number_list()
    while True:
        image, box = generate_single_image(img_list)
        x, shape = process_input_image(image, model.input_dim, process_method)
        x = np.expand_dims(x, axis=0)
        result = predict_model.predict(x)
        result_list = decode_result(result, model_name=model)
        draw_image(image, result_list, process_method, model.input_dim)


if __name__ == '__main__':
    a = []
    with open("./data/nature_annotation_for_prediction.txt", 'r') as f:
        for line in f:
            a.append(line)
    prediction_for_recording(record_name="simple_mobilenetv2_default_prior_box_all_generated",
                             weight_file=[
                                 get_weight_file('SSD_ep030-loss0.239-val_loss0.130-ModelConfig.MobileNetV2.h5'), ],
                             model_name=ModelConfig.MobileNetV2,
                             use_generator=True,
                             generator_count=200,
                             use_annotation=False,
                             annotation_lines=a,
                             save_image=False,
                             method=PMethod.Reshape)
