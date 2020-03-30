import os
import numpy as np
import xml.dom.minidom
from PIL import Image
import random


def img_extraction_generator(file_path):
    annotation = xml.dom.minidom.parse(file_path).documentElement
    path = annotation.getElementsByTagName("path")
    image_full_path = path[0].childNodes[0].data
    object_list = annotation.getElementsByTagName('object')
    result_list = []
    for o in object_list:
        net = {"name": int(o.getElementsByTagName(
            "name")[0].childNodes[0].data)}
        bnd = o.getElementsByTagName("bndbox")
        for bound in bnd:
            x_min = int(bound.getElementsByTagName(
                'xmin')[0].childNodes[0].data)
            x_max = int(bound.getElementsByTagName(
                'xmax')[0].childNodes[0].data)
            y_min = int(bound.getElementsByTagName(
                'ymin')[0].childNodes[0].data)
            y_max = int(bound.getElementsByTagName(
                'ymax')[0].childNodes[0].data)
            net["xmin"] = x_min
            net["xmax"] = x_max
            net["ymin"] = y_min
            net["ymax"] = y_max
        result_list.append(net)
    return image_full_path, result_list


def generate_annotation(voc_path):
    target_path = "./data/real_annotation.txt"
    if os.path.exists(target_path):
        print("File already exists.")
        return
    with open(target_path, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(voc_path):
            for f in files:
                full_path = os.path.join(root, f)
                if full_path.endswith("xml"):
                    image_path, result = img_extraction_generator(full_path)
                    out.write(os.path.join("G:/data_stored/nature_train", image_path.split("\\")[-1]))
                    for k in result:
                        out.write(" {},{},{},{},{}".format(k['xmin'], k['ymin'], k['xmax'], k['ymax'], k['name']))
                    out.write("\n")


def rect_cross(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max):
    zx = abs(x1_min + x1_max - x2_min - x2_max)
    x = abs(x1_min - x1_max) + abs(x2_min - x2_max)
    zy = abs(y1_min + y1_max - y2_min - y2_max)
    y = abs(y1_min - y1_max) + abs(y2_min - y2_max)
    if zx <= x and zy <= y:
        return 1
    else:
        return 0


def generate_single_image(file_list):
    while True:
        number_list = []
        # random height
        image_height = random.randint(50, 80)
        # random width
        image_width = int((1 + random.random()) * 5 * image_height)
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
        for i in range(0, 7):
            pick_num = random.randint(0, 9)
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


def random_image_generator(digit_dir, save_path, voc_path):
    digit_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    digit_list = []
    for i in digit_class:
        single_digit_list = []
        path = os.path.join(digit_dir, i)
        for root, dirs, files in os.walk(path):
            for digit_file in files:
                full_path = os.path.join(path, digit_file)
                single_digit_list.append(full_path)
        digit_list.append(single_digit_list)
    if os.path.exists(voc_path):
        print("VOC file already exists.")
        return
    fout = open(voc_path, "w", encoding="utf-8")
    for i in range(0, 3000):
        img, l = generate_single_image(digit_list)
        s = os.path.join(save_path, "{:0>5d}.jpg".format((i + 1)))
        img.save(s)
        fout.write(s)
        for k in l:
            fout.write(" {},{},{},{},{}".format(k['xmin'], k['ymin'], k['xmax'], k['ymax'], k['class']))
        fout.write("\n")
        if (i + 1) % 50 == 0:
            print("finish {} images.".format(i))
    fout.close()
