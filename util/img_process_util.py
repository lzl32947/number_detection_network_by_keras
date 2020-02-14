from PIL import Image
import numpy as np
from parameter.parameters import HyperParameter


def process_single_input(img_path):
    img = Image.open(img_path)
    # change the image to numpy array with 3 dimension
    img_shape = np.array(np.shape(img)[0:2])
    width = img_shape[1]
    height = img_shape[0]
    # 计算缩放比例(相对于原图片)
    width_zoom_ratio = HyperParameter.min_dim / width
    height_zoom_ratio = HyperParameter.min_dim / height
    # 确定缩放比例
    zoom_ratio = min(width_zoom_ratio, height_zoom_ratio)
    # 调整图像
    new_width = int(zoom_ratio * width)
    new_height = int(zoom_ratio * height)
    # 注意resize的图像应该是先宽后高，这是Image库resize的定义
    img_t = img.resize((new_width, new_height), Image.BICUBIC)
    # 新建图像并居中复制
    new_img = Image.new('RGB', (HyperParameter.min_dim, HyperParameter.min_dim), (128, 128, 128))
    width_offset = (HyperParameter.min_dim - new_width) // 2
    height_offset = (HyperParameter.min_dim - new_height) // 2
    new_img.paste(img_t, (width_offset, height_offset))
    # 计算偏移量
    x_offset_ratio = width_offset / HyperParameter.min_dim
    y_offset_ratio = height_offset / HyperParameter.min_dim
    # 转换为numpy数组
    photo = np.array(new_img, dtype=np.float64)
    # 优化图像
    photo = np.reshape(photo, [1, HyperParameter.min_dim, HyperParameter.min_dim, 3])
    photo = process_pixel(photo)
    return photo, x_offset_ratio, y_offset_ratio, img


def process_pixel(img_array):
    # 该函数旨在简化keras中preprocess_input函数的工作过程
    # 和preprocess_input函数返回相同的值
    # 'RGB'->'BGR'
    img_array = img_array[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]
    return img_array
