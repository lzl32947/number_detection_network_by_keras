import os
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from parameter.parameters import HyperParameter
from util.img_process_util import process_single_input


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
                    net = {"name": int(o.getElementsByTagName("name")[0].childNodes[0].data)}
                    bnd = o.getElementsByTagName("bndbox")
                    for bound in bnd:
                        x_min = int(bound.getElementsByTagName('xmin')[0].childNodes[0].data)
                        x_max = int(bound.getElementsByTagName('xmax')[0].childNodes[0].data)
                        y_min = int(bound.getElementsByTagName('ymin')[0].childNodes[0].data)
                        y_max = int(bound.getElementsByTagName('ymax')[0].childNodes[0].data)
                        net["xmin"] = x_min
                        net["xmax"] = x_max
                        net["ymin"] = y_min
                        net["ymax"] = y_max
                    result_list.append(net)

                yield image_full_path, result_list
            else:
                continue


def raw_data_generator(dir_path):
    for path, img_bound_list in img_extraction_generator(dir_path):
        path = os.path.join(dir_path, path.split("\\")[-1])
        photo, x_offset_ratio, y_offset_ratio, img = process_single_input(path)
        img_shape = np.shape(img)
        x_range = HyperParameter.min_dim / img_shape[1]
        y_range = HyperParameter.min_dim / img_shape[0]
        ranges = min(x_range, y_range)
        for bound_box_dict in img_bound_list:
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
        yield photo, img_bound_list
