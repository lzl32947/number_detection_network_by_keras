import os
import pickle

from keras import Model
from keras.utils import plot_model

from config.Configs import Config, PMethod
from model.ResNet50_base import ResNet50, ssd_network
from models import load_config
from util.data_util import data_generator

if __name__ == '__main__':
    # process_method = PMethod.Zoom
    # model = choose_model("vgg16", 20)
    # model.load_weights(os.path.join(Config.weight_dir, "ssd_weights_20.h5"), by_name=True, skip_mismatch=True)
    # image_path = os.path.join(Config.test_data_dir, "street2.jpg")
    # x, shape = process_input_image(image_path, process_method)
    # result = model.predict(x)
    # result_list = decode_result(result)
    # draw_image(image_path,result_list,process_method)
    # p = pickle.load(open(os.path.join(Config.prior_box_dir, "prior_boxes_ssd300.pkl"), 'rb'))
    # print(p)
    # Config.priors = p
    # u = (30, 30, 40, 50)
    # calculate_max_iou(np.array(u))
    # count = 0
    # for x, y in data_generator(Config.train_annotation_path, batch_size=4, method=PMethod.Reshape):
    #     count += 1
    #     if count % 10 == 0:
    #         print(count)
    # generate_annotation(r"G:\data_stored\train_voc")
    # random_image_generator("./data/digits", save_path="./data/generated_image", voc_path="./data/generated_image.txt")
    # load_config("vgg16", 10)
    # resnet = ResNet50()
    # net = ssd_network(resnet)
    # model = Model(inputs=resnet.input, outputs=net['predictions'])
    # model.summary()
    # plot_model(model, to_file="resnet50.png", show_layer_names=True, show_shapes=True)
    pass