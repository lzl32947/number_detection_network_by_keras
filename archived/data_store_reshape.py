import os
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def img_extraction_generator(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for xml_file in files:
            if xml_file.endswith("xml"):
                file_name = os.path.join(root, xml_file)
                annotation = xml.dom.minidom.parse(file_name).documentElement
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

                yield image_full_path, result_list
            else:
                continue


if __name__ == '__main__':
    BASE_SIZE = 300
    txt = open("G:\\data_stored\\train.txt", "w", encoding="utf-8")

    for root, dirs, files in os.walk("G:\\data_stored\\reshape_train"):
        for file in files:
            path = os.path.join(root, file)
            os.remove(path)

    for img, res in img_extraction_generator("G:\\data_stored\\train_voc"):
        img_name = img.split("\\")[-1]
        img = os.path.join("G:\\data_stored\\line", img_name)
        print("opening image {}.".format(img), end=" ")
        image = Image.open(img)
        width = np.array(image).shape[1]
        height = np.array(image).shape[0]
        image = image.resize((BASE_SIZE, BASE_SIZE), Image.ANTIALIAS)

        plt_array = np.array(image)
        if plt_array.shape[2] == 4:
            plt_array = plt_array[:, :, 0:3]
        print(plt_array.shape)
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        plt.imshow(plt_array)

        x_ratio = BASE_SIZE / width
        y_ratio = BASE_SIZE / height

        save_path = os.path.join("G:\\data_stored\\reshape_train", img_name)
        image = Image.fromarray(plt_array)
        image.save(save_path)
        txt.write(save_path)

        for digits in res:
            x_min = digits['xmin']
            y_min = digits['ymin']
            x_max = digits['xmax']
            y_max = digits['ymax']
            num = digits['name']

            x_min = int(x_min * x_ratio)
            x_max = int(x_max * x_ratio)
            y_min = int(y_min * y_ratio)
            y_max = int(y_max * y_ratio)

            rec = plt.Rectangle((x_min, y_min), x_max - x_min, y_max -
                                y_min, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rec)

            txt.write(" {},{},{},{},{}".format(x_min, y_min, x_max, y_max, num))

        txt.write("\n")
        plt.show()
        plt.close()
    txt.close()
