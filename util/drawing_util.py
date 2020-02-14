import matplotlib.pyplot as plt
import cv2
from PIL import ImageFont
import numpy as np
from PIL.ImageDraw import Draw

from parameter.parameters import HyperParameter


def draw_image(image, input_list, offset_x=0, offset_y=0):
    draw = Draw(image)
    for item in input_list:
        label = item['class']
        for (box, conf) in zip(item['box'], item['conf']):
            p1_x = box[0]
            p1_y = box[1]
            p2_x = box[2]
            p2_y = box[3]
            img_height = np.shape(image)[0]
            img_width = np.shape(image)[1]
            x_min, y_min, x_max, y_max = correct_box(p1_x, p1_y, p2_x, p2_y, img_width, img_height,
                                                     offset_x, offset_y)
            draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 0, 0))
    image.show()


def correct_box(x_min, y_min, x_max, y_max, iw, ih, x_offset_ratio=0, y_offset_ratio=0):
    # 计算坐标相对于300,300图像的位置
    x_min *= HyperParameter.min_dim
    y_min *= HyperParameter.min_dim
    x_max *= HyperParameter.min_dim
    y_max *= HyperParameter.min_dim
    # 根据宽高比例确定是否进行了某一维度的缩放
    if x_offset_ratio > 0:
        # 调整x坐标到300,300图像的相应位置
        if x_min < HyperParameter.min_dim * 0.5:
            x_min = HyperParameter.min_dim * 0.5 - ((HyperParameter.min_dim * 0.5 - x_min) * ih / iw)
        else:
            x_min = HyperParameter.min_dim * 0.5 + (x_min - HyperParameter.min_dim * 0.5) * ih / iw
        if x_max < HyperParameter.min_dim * 0.5:
            x_max = HyperParameter.min_dim * 0.5 - ((HyperParameter.min_dim * 0.5 - x_max) * ih / iw)
        else:
            x_max = HyperParameter.min_dim * 0.5 + (x_max - HyperParameter.min_dim * 0.5) * ih / iw
    if y_offset_ratio > 0:
        # 调整y坐标到300,300图像的相应位置
        if y_min < HyperParameter.min_dim * 0.5:
            y_min = HyperParameter.min_dim * 0.5 - ((HyperParameter.min_dim * 0.5 - y_min) * iw / ih)
        else:
            y_min = HyperParameter.min_dim * 0.5 + (y_min - HyperParameter.min_dim * 0.5) * iw / ih
        if y_max < HyperParameter.min_dim * 0.5:
            y_max = HyperParameter.min_dim * 0.5 - ((HyperParameter.min_dim * 0.5 - y_max) * iw / ih)
        else:
            y_max = HyperParameter.min_dim * 0.5 + (y_max - HyperParameter.min_dim * 0.5) * iw / ih
    # 将坐标等比例放大成为原图像以避免出现失真问题
    x_min = x_min / HyperParameter.min_dim * iw
    x_max = x_max / HyperParameter.min_dim * iw
    y_min = y_min / HyperParameter.min_dim * ih
    y_max = y_max / HyperParameter.min_dim * ih
    return x_min, y_min, x_max, y_max
