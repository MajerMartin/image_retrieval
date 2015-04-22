__author__ = 'martin.majer'

import cv2

def keep_ratio(shape, height, width):
    '''
    Calculates dimensions for resizing without affecting ratio.
    :param shape: shape of array of image saved in numpy
    :param height: height the image will be resized to
    :param width: width the image will be resized to
    :return: dimensions the image will be resized to
    '''
    # height <= width
    if shape[0] <= shape[1]:
        ratio = height / shape[0]
        width = shape[1] * ratio
    else:
        ratio = width / shape[1]
        height = shape[0] * ratio

    # opencv dimension format
    dim = int(width), int(height)

    return dim

def crop(img, height, width):
    '''
    Crop center of the image.
    :param img: image to be cropped
    :param height: height the image will be cropped to
    :param width: width the image will be cropped to
    :return: cropped image
    '''
    if (img.shape[0] == height) and (img.shape[1] == width):
        return img
    elif img.shape[0] == height:
        middle = img.shape[1] / 2
        return img[:, (middle - width / 2):(middle + width / 2)]
    else:
        middle = img.shape[0] / 2
        return img[(middle - height / 2):(middle + height / 2), :]

def resize_crop(img, height, width):
    '''
    Resize and crop image while keeping ratio.
    :param img: image to be resized and cropped
    :param height: height the image will be cropped to
    :param width: width the image will be cropped to
    :return: resized and cropped image
    '''
    dim = keep_ratio(img.shape, height, width)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    img_cropped = crop(img_resized, height, width)

    return img_cropped