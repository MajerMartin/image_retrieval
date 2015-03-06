__author__ = 'martin.majer'

import numpy as np
import cv2


class ImageSearchDistanceMatrix(object):
    def __init__(self, max_images=100000, thumbnail_size=(150,150,3)):
        self.max_images = max_images
        self.thumbnail_size = thumbnail_size
        self.distance_matrix = None
        self.images = []
        self.features = []

    def add_images(self, images, features):
        # add features
        if len(features.shape) == 1:
            self.features.append(features)
            start = len(self.features) - 1
        else:
            self.features.extend(features)
            start = len(self.features) - len(features)

        dim = (self.thumbnail_size[0], self.thumbnail_size[1])

        # add resized images
        if len(images.shape) == 3:
            img_resized = cv2.resize(images, dim, interpolation = cv2.INTER_NEAREST)  # INTER_CUBIC changes pixel values
            self.images.append(img_resized)

        else:
            for img in images:
                img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
                self.images.append(img_resized)

        end = len(self.features)

        # initialize new distance matrix
        if self.distance_matrix is None:
            self.distance_matrix = np.zeros((end, end))

            # fill new distance matrix
            for i in range(0, end):
                for j in range(i, end):
                    self.distance_matrix[i,j] = np.linalg.norm(self.features[i] - self.features[j])

        # add rows and columns
        else:
            for i in range(start, end):
                new_row = np.zeros((1, i))
                self.distance_matrix = np.concatenate((self.distance_matrix, new_row), axis=0)

                new_col = np.zeros((i + 1, 1))

                for j in range(0, i):
                    new_col[j,0] = np.linalg.norm(self.features[i] - self.features[j])

                self.distance_matrix = np.concatenate((self.distance_matrix, new_col), axis=1)

    def find_k_nearest_by_index(self, img_index, k=3):
        pass

    def get_images(self, indexes):
        images = []

        for index in indexes:
            images.append(self.images[index])

        return images

    def save(self, filename):
        # ulozit do H5, klice images, features, max_images, thumbnail_size, distance_matrix (ulozit jen spodni trojuhelnik)
        pass

    def load(self, filename):
        pass