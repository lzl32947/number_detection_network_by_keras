from PIL.ImageDraw import Draw

from Configs import Config, PMethod
from PIL import Image
import numpy as np


def resize_image(image):
    new_img = image.resize((Config.input_dim, Config.input_dim), Image.ANTIALIAS)
    return new_img


def zoom_image(image):
    # change the image to numpy array with 3 dimension
    img_shape = np.array(np.shape(image)[0:2])
    width = img_shape[1]
    height = img_shape[0]
    # 计算缩放比例(相对于原图片)
    width_zoom_ratio = Config.input_dim / width
    height_zoom_ratio = Config.input_dim / height
    # 确定缩放比例
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    # 调整图像
    new_width = int(zoom_ratio * width)
    new_height = int(zoom_ratio * height)
    # 注意resize的图像应该是先宽后高，这是Image库resize的定义
    img_t = image.resize((new_width, new_height), Image.BICUBIC)
    # 新建图像并居中复制
    new_img = Image.new('RGB', (Config.input_dim, Config.input_dim), (128, 128, 128))

    width_offset = (Config.input_dim - new_width) // 2
    height_offset = (Config.input_dim - new_height) // 2
    new_img.paste(img_t, (width_offset, height_offset))
    return new_img


def draw_image(image_path, result_list, method):
    image = Image.open(image_path)
    shape = np.array(image).shape
    draw = Draw(image)
    width_zoom_ratio = Config.input_dim / shape[1]
    height_zoom_ratio = Config.input_dim / shape[0]
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    new_width = int(zoom_ratio * shape[1])
    new_height = int(zoom_ratio * shape[0])

    width_offset = (Config.input_dim - new_width) // 2
    height_offset = (Config.input_dim - new_height) // 2
    width_offset /= Config.input_dim
    height_offset /= Config.input_dim

    for box, conf, index in result_list:
        if method == PMethod.Reshape:
            x_min = shape[1] * box[0]
            y_min = shape[0] * box[1]
            x_max = shape[1] * box[2]
            y_max = shape[0] * box[3]
        else:
            x_min = Config.input_dim * box[0]
            y_min = Config.input_dim * box[1]
            x_max = Config.input_dim * box[2]
            y_max = Config.input_dim * box[3]
            if width_offset > 0:
                if x_min < Config.input_dim * 0.5:
                    x_min = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - x_min) * shape[0] / shape[1])
                else:
                    x_min = Config.input_dim * 0.5 + (x_min - Config.input_dim * 0.5) * shape[0] / shape[1]
                if x_max < Config.input_dim * 0.5:
                    x_max = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - x_max) * shape[0] / shape[1])
                else:
                    x_max = Config.input_dim * 0.5 + (x_max - Config.input_dim * 0.5) * shape[0] / shape[1]
            if height_offset > 0:
                if y_min < Config.input_dim * 0.5:
                    y_min = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - y_min) * shape[1] / shape[0])
                else:
                    y_min = Config.input_dim * 0.5 + (y_min - Config.input_dim * 0.5) * shape[1] / shape[0]
                if y_max < Config.input_dim * 0.5:
                    y_max = Config.input_dim * 0.5 - ((Config.input_dim * 0.5 - y_max) * shape[1] / shape[0])
                else:
                    y_max = Config.input_dim * 0.5 + (y_max - Config.input_dim * 0.5) * shape[1] / shape[0]
            x_min = x_min / Config.input_dim * shape[1]
            x_max = x_max / Config.input_dim * shape[1]
            y_min = y_min / Config.input_dim * shape[0]
            y_max = y_max / Config.input_dim * shape[0]

        draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 0, 0))
    image.show()
