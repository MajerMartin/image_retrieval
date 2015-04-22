__author__ = 'martin.majer'

import unittest
import os
import cv2

import img_search.images

images = img_search.images

height = 227.
width = 227.

storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'images_test')

vertical = cv2.imread(os.path.join(storage_dir, 'vertical.jpg'))
horizontal = cv2.imread(os.path.join(storage_dir, 'horizontal.jpg'))
square = cv2.imread(os.path.join(storage_dir, 'square.jpg'))

class ImagesTestClass(unittest.TestCase):

    def test_keep_ratio(self):
        vertical_dim = images.keep_ratio(vertical.shape, height, width)
        self.assertEqual(vertical_dim[0], height)

        horizontal_dim = images.keep_ratio(horizontal.shape, height, width)
        self.assertEqual(horizontal_dim[1], height)

        square_dim = images.keep_ratio(square.shape, height, width)
        self.assertEqual(square_dim, (height, height))

    def test_resize_crop(self):
        vertical_cropped = images.resize_crop(vertical, height, width)
        self.assertEqual(vertical_cropped.shape, (height, height, 3))

        horizontal_cropped = images.resize_crop(horizontal, height, width)
        self.assertEqual(horizontal_cropped.shape, (height, height, 3))

        square_cropped = images.resize_crop(square, height, width)
        self.assertEqual(square_cropped.shape, (height, height, 3))