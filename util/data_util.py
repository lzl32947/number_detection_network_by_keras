import pickle
import random
from PIL import Image
import numpy as np
import os
from parameter.parameters import HyperParameter, TrainParameter, DataParameter
from util.img_process_util import process_single_input
import matplotlib.pyplot as plt


def generate_label(box_list, priors):
    """
    This function is to generate label for each image.
    :param box_list: list of box-dict with keys(class,xmin,ymin,xmax,ymax)
    :param priors: the pre-loaded prior boxes with shape (8732,4)
    :return: the label of the image with shape (8732,class_num + 12), in which contains 4 prior boxes, 4 variances and
    4 locations.
    """
    result_array = np.zeros(shape=(priors.shape[0], HyperParameter.class_num + 12), dtype="float32")
    # all background
    result_array[:, 4] = 1.0

    # [0:4] : loc
    # [4:-8] : softmax result
    # [-8:-4] : prior boxes, which can be set to 0 since the loss function will not calculate this
    # Attention : 在loss function中将-8位作为计数位用来判断该图像中的正样本数量，因此有样本的这一位需要置1
    # [-4:] : variance, which can set to 0 for not calculating this

    for digit in box_list:
        digit_class = 1 + digit['class']
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
        result_array[index, -8] = 1.0
    return result_array


def calculate_max_iou(priors_array, x_min, y_min, x_max, y_max):
    """
    This function calculate the iou with each prior box and return the index of the selected box.
    :param priors_array: the pre-loaded prior boxes with shape(8732,4)
    :param x_min: x value of left-top point
    :param y_min: y value of left-top point
    :param x_max: x value of right-bottom point
    :param y_max: y value of right-bottom point
    :return: index of the prior box which has the max iou
    """
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


def adjust_boxes(raw_image, box_list, reshape_only=DataParameter.reshape_only):
    """
    This function is to adjust box to fit the image
    :param reshape_only: whether just reshape the image without encoding
    :param raw_image: the Image object or numpy array, with shape(?,?,3)
    :param box_list: the list of box-dict with keys(class,xmin,ymin,xmax,ymax)
    :return: the adjusted box-dict list
    """

    img_shape = np.shape(raw_image)
    x_range = HyperParameter.min_dim / img_shape[1]
    y_range = HyperParameter.min_dim / img_shape[0]
    if reshape_only:
        for bound_box_dict in box_list:
            bound_box_dict['xmin'] = bound_box_dict['xmin'] * x_range
            bound_box_dict['xmax'] = bound_box_dict['xmax'] * x_range
            bound_box_dict['ymin'] = bound_box_dict['ymin'] * y_range
            bound_box_dict['ymax'] = bound_box_dict['ymax'] * y_range
    else:
        # Attention,这个函数是解码过程的逆向过程,which is not simply zooming in.
        ranges = min(x_range, y_range)
        for bound_box_dict in box_list:
            if bound_box_dict["xmin"] < img_shape[1] / 2:
                x_min = int(HyperParameter.min_dim / 2 - (img_shape[1] / 2 - bound_box_dict["xmin"]) * ranges) - 1
                bound_box_dict["xmin"] = x_min
            else:
                x_min = int(HyperParameter.min_dim / 2 + (bound_box_dict["xmin"] - img_shape[1] / 2) * ranges) - 1
                bound_box_dict["xmin"] = x_min
            if bound_box_dict["xmax"] < img_shape[1] / 2:
                x_max = int(HyperParameter.min_dim / 2 - (img_shape[1] / 2 - bound_box_dict["xmax"]) * ranges) + 1
                bound_box_dict["xmax"] = x_max
            else:
                x_max = int(HyperParameter.min_dim / 2 + (bound_box_dict["xmax"] - img_shape[1] / 2) * ranges) + 1
                bound_box_dict["xmax"] = x_max
            if bound_box_dict["ymin"] < img_shape[0] / 2:
                y_min = int(HyperParameter.min_dim / 2 - (img_shape[0] / 2 - bound_box_dict["ymin"]) * ranges) - 1
                bound_box_dict["ymin"] = y_min
            else:
                y_min = int(HyperParameter.min_dim / 2 + (bound_box_dict["ymin"] - img_shape[0] / 2) * ranges) - 1
                bound_box_dict["ymin"] = y_min
            if bound_box_dict["ymax"] < img_shape[0] / 2:
                y_max = int(HyperParameter.min_dim / 2 - (img_shape[0] / 2 - bound_box_dict["ymax"]) * ranges) + 1
                bound_box_dict["ymax"] = y_max
            else:
                y_max = int(HyperParameter.min_dim / 2 + (bound_box_dict["ymax"] - img_shape[0] / 2) * ranges) + 1
                bound_box_dict["ymax"] = y_max
    return box_list


def rect_cross(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max):
    """
    This function calculate whether the two rectangle cross over each other.
    :return: the two rectangles have inner set
    """
    zx = abs(x1_min + x1_max - x2_min - x2_max)
    x = abs(x1_min - x1_max) + abs(x2_min - x2_max)
    zy = abs(y1_min + y1_max - y2_min - y2_max)
    y = abs(y1_min - y1_max) + abs(y2_min - y2_max)
    if zx <= x and zy <= y:
        return 1
    else:
        return 0


def generate_array_and_label(image, number_list, priors):
    """
    This function return the image and label of the train data.
    :param image: Image object
    :param number_list: the position of boxes
    :param priors: the pre-loaded prior array
    :return: the train array and the label
    """
    # adjust boxes, remember this is the reversed process of decode function.
    # e.g. To centralized, make a new image of (300,300,3), paste and adjust box.
    k = adjust_boxes(image, number_list)
    # process the input image, mainly change type to float64 and standardized it.
    x = process_single_input(image)[0]
    # whether to show the image
    if DataParameter.show_image:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(x)
        for i in k:
            xmin = i['xmin']
            ymin = i['ymin']
            xmax = i['xmax']
            ymax = i['ymax']
            rec = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                color='red')
            ax.add_patch(rec)
        plt.show()
        plt.close()
    # generate label of image
    y = generate_label(k, priors)
    return x, y


# The following are the items in dict
# image_list:[]
# -> image_dict:{path,pos}
# -> image_dict -> pos : []
# -> image_dict -> pos -> dict{xmin,xmax,ymin,ymax,class}

def generate_single_image(file_list):
    """
    This function create a single generated image instead of a real one for training.
    :param file_list: the list contains the single number.
    :param priors: the pre-loaded prior boxes with shape(8732,4)
    :return: the generated numpy array and its label
    """
    while True:
        number_list = []
        # random height
        image_height = random.randint(TrainParameter.image_height[0], TrainParameter.image_height[1])
        # random width
        image_width = int(
            (TrainParameter.image_ratio[0] + random.random()) * TrainParameter.image_ratio[1] * image_height)
        # add_image
        new_img = Image.new('RGB', (image_width, image_height), (128, 128, 128))
        new_img = np.array(new_img)
        # add noise
        sigma = 25
        r = new_img[:, :, 0].flatten()
        g = new_img[:, :, 1].flatten()
        b = new_img[:, :, 2].flatten()
        for point in range(new_img.shape[0] * new_img.shape[1]):
            r[point] = r[point] + random.gauss(0, sigma)
            g[point] = g[point] + random.gauss(0, sigma)
            b[point] = b[point] + random.gauss(0, sigma)
        new_img[:, :, 0] = r.reshape([new_img.shape[0], new_img.shape[1]])
        new_img[:, :, 1] = g.reshape([new_img.shape[0], new_img.shape[1]])
        new_img[:, :, 2] = b.reshape([new_img.shape[0], new_img.shape[1]])
        new_img = Image.fromarray(new_img)
        # Add number in that
        for i in range(0, TrainParameter.max_num):
            pick_num = random.randint(0, len(TrainParameter.class_list) - 1)
            index = random.randint(0, len(file_list[pick_num]) - 1)
            # open this
            number_image = Image.open(file_list[pick_num][index])
            # calculate shape
            width = np.shape(number_image)[1]
            height = np.shape(number_image)[0]
            if len(number_list) > 0:
                # the list is not empty
                # define times for trail, after which the picked number should be abandon
                try_time = 10
                while try_time > 0:
                    x_min = random.randint(0, image_width - width - 1)
                    y_min = random.randint(0, image_height - height - 1)
                    x_max = x_min + width
                    y_max = y_min + height
                    # if conflict with pre-generated number, then abandon this
                    flag_conflict = False
                    for item_dict in number_list:
                        if rect_cross(x_min, y_min, x_max, y_max, item_dict['xmin'], item_dict['ymin'],
                                      item_dict['xmax'], item_dict["ymax"]):
                            flag_conflict = True
                            break
                    # not conflict, add this to list
                    if not flag_conflict:
                        number_dict = {
                            "xmin": x_min,
                            "ymin": y_min,
                            "xmax": x_max,
                            "ymax": y_max,
                            "class": pick_num
                        }
                        new_img.paste(number_image, (x_min, y_min))
                        number_list.append(number_dict)
                        break
                    # conflict, abandon this
                    else:
                        try_time -= 1

                if try_time == 0:
                    # go for picking another number and retry this.
                    continue

            else:
                # the list is empty and can directly add that into list
                x_min = random.randint(0, image_width - width - 1)
                y_min = random.randint(0, image_height - height - 1)
                x_max = x_min + width
                y_max = y_min + height
                number_dict = {
                    "xmin": x_min,
                    "ymin": y_min,
                    "xmax": x_max,
                    "ymax": y_max,
                    "class": pick_num
                }
                new_img.paste(number_image, (x_min, y_min))
                number_list.append(number_dict)
        break
    return new_img, number_list


def get_image_number_list():
    f_list = []
    for class_name in TrainParameter.class_list:
        single_digit_list = []
        for root, dirs, files in os.walk(os.path.join(TrainParameter.single_data_path, class_name)):
            for file in files:
                full_path = os.path.join(root, file)
                single_digit_list.append(full_path)
        f_list.append(single_digit_list)
    return f_list


def data_generator(batch_size, train, priors, validation_ratio=0.8, real_image_ratio=0.1):
    # prepare real images
    real_image_list = []
    with open(TrainParameter.voc_path, "r") as fin:
        for line in fin:
            img_dict = {}
            single_line = line.split(" ")
            img_dict["path"] = single_line[0]
            positions = []
            for i in range(1, len(single_line)):
                position_dict = {}
                position_list = single_line[i].split(",")
                position_dict["xmin"] = int(position_list[0])
                position_dict["ymin"] = int(position_list[1])
                position_dict["xmax"] = int(position_list[2])
                position_dict["ymax"] = int(position_list[3])
                position_dict["class"] = int(position_list[4])
                positions.append(position_dict)
            img_dict["pos"] = positions
            real_image_list.append(img_dict)
    random.shuffle(real_image_list)
    point = int(validation_ratio * len(real_image_list))

    # prepare generated images
    file_list = get_image_number_list()
    counter = 0
    X = []
    Y = []
    while counter < batch_size:
        if random.random() < real_image_ratio:
            if train:
                index = random.randint(0, point - 1)
            else:
                index = random.randint(point, len(real_image_list) - 1)
            img = Image.open(real_image_list[index]['path'])
            x, y = generate_array_and_label(img, real_image_list[index]['pos'], priors)
            X.append(x)
            Y.append(y)
        else:
            img, file_list = generate_single_image(file_list)
            x, y = generate_array_and_label(img, file_list, priors)
            X.append(x)
            Y.append(y)
        counter += 1
        if counter == batch_size:
            counter = 0
            yield np.array(X), np.array(Y)
            X = []
            Y = []


def read_from_file_generator(batch_size, priors, split_ratio=0.9, train=True):
    # prepare real images
    real_image_list = []
    # with open(TrainParameter.voc_path, "r") as fin:
    #     for line in fin:
    #         img_dict = {}
    #         single_line = line.split(" ")
    #         img_dict["path"] = single_line[0]
    #         positions = []
    #         for i in range(1, len(single_line)):
    #             position_dict = {}
    #             position_list = single_line[i].split(",")
    #             position_dict["xmin"] = int(position_list[0])
    #             position_dict["ymin"] = int(position_list[1])
    #             position_dict["xmax"] = int(position_list[2])
    #             position_dict["ymax"] = int(position_list[3])
    #             position_dict["class"] = int(position_list[4])
    #             positions.append(position_dict)
    #         img_dict["pos"] = positions
    #         real_image_list.append(img_dict)
    with open(TrainParameter.generated_voc, "r") as fin:
        for line in fin:
            img_dict = {}
            single_line = line.split(" ")
            img_dict["path"] = single_line[0]
            positions = []
            for i in range(1, len(single_line)):
                position_dict = {}
                position_list = single_line[i].split(",")
                position_dict["xmin"] = int(position_list[0])
                position_dict["ymin"] = int(position_list[1])
                position_dict["xmax"] = int(position_list[2])
                position_dict["ymax"] = int(position_list[3])
                position_dict["class"] = int(position_list[4])
                positions.append(position_dict)
            img_dict["pos"] = positions
            real_image_list.append(img_dict)
    random.shuffle(real_image_list)

    point = int(split_ratio * len(real_image_list))
    counter = 0
    X = []
    Y = []
    while counter < batch_size:
        if train:
            index = random.randint(0, point - 1)
        else:
            index = random.randint(point, len(real_image_list) - 1)
        img = Image.open(real_image_list[index]['path'])
        x, y = generate_array_and_label(img, real_image_list[index]['pos'], priors)
        X.append(x)
        Y.append(y)
        counter += 1
        if counter == batch_size:
            counter = 0
            yield np.array(X), np.array(Y)
            X = []
            Y = []


def generate_image():
    """
    This function generate train data and save it to files.
    :return: None
    """
    for root, dirs, files in os.walk(TrainParameter.data_store_path):
        for img in files:
            os.remove(os.path.join(root, img))

    f = open(TrainParameter.generated_voc, "w")
    file_list = get_image_number_list()
    for j in range(0, TrainParameter.max_generate_count):
        image, raw_box_list = generate_single_image(file_list)
        box_list = adjust_boxes(image, raw_box_list, True)
        DataParameter.process_pixel = False
        m = Image.fromarray(process_single_input(image)[0])
        if DataParameter.show_image:
            save_path = os.path.join(TrainParameter.data_store_path, "{:0>5d}.jpg".format(j))
            m.save(save_path)
            f.write(save_path)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(m)
            for i in box_list:
                xmin = int(i['xmin'])
                ymin = int(i['ymin'])
                xmax = int(i['xmax'])
                ymax = int(i['ymax'])
                num = i['class']
                f.write(" {},{},{},{},{}".format(xmin, ymin, xmax, ymax, num))
                rec = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                    color='red')
                ax.add_patch(rec)
            f.write("\n")
            plt.show()
            plt.close()
        else:
            save_path = os.path.join(TrainParameter.data_store_path, "{:0>5d}.jpg".format(j))
            m.save(save_path)
            f.write(save_path)
            for i in box_list:
                xmin = int(i['xmin'])
                ymin = int(i['ymin'])
                xmax = int(i['xmax'])
                ymax = int(i['ymax'])
                num = i['class']
                f.write(" {},{},{},{},{}".format(xmin, ymin, xmax, ymax, num))
            f.write("\n")
        if j % 100 == 0:
            print("finish {} image.".format(j))
    f.close()
