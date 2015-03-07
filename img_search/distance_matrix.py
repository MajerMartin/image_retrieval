__author__ = 'martin.majer'

import numpy as np
import cv2
import h5py


class ImageSearchDistanceMatrix(object):
    def __init__(self, max_images=100000, thumbnail_size=(150,150,3)):
        self.max_images = max_images
        self.thumbnail_size = thumbnail_size
        self.distance_matrix = None
        self.images = []
        self.features = []

    def add_images(self, images, features):
        '''
        Add images, their features and calculate distance matrix.
        :param images: list of images
        :param features: list of features
        :return: nothing
        '''
        dim = (self.thumbnail_size[0], self.thumbnail_size[1])

        # add resized images
        if len(images.shape) == 3:
            if (len(self.images) + 1) > self.max_images:
                print 'You can add only %d more image(s). Maximum limit achieved.' % (self.max_images - len(self.images))
                return
            else:
                img_resized = cv2.resize(images, dim, interpolation = cv2.INTER_NEAREST)  # INTER_CUBIC changes pixel values
                self.images.append(img_resized)
        else:
            if (len(self.images) + len(images)) > self.max_images:
                print 'You can add only %d more image(s). Maximum limit achieved.' % (self.max_images - len(self.images))
                return
            else:
                for img in images:
                    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
                    self.images.append(img_resized)

        # add features
        if len(features.shape) == 1:
            self.features.append(features)
            start = len(self.features) - 1
        else:
            self.features.extend(features)
            start = len(self.features) - len(features)

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

        self.distance_matrix += self.distance_matrix.T

    def find_k_nearest_by_index(self, img_index, k=3):
        '''
        Find K nearest neighbors by image index.
        :param img_index: index of searched image
        :param k: neighbors count
        :return: k nearest neighbors
        '''
        row = self.distance_matrix[img_index,:]
        nearest = np.argsort(row)

        return nearest[:k]

    def get_images(self, indexes):
        '''
        Get images on corresponding indexes.
        :param indexes: indexes of images
        :return: images
        '''
        images = []

        for index in indexes:
            images.append(self.images[index])

        return images

    def save(self, filename):
        '''
        Save object variables to HDF5.
        :param filename: name of HDF5 file
        :return: nothing
        '''
        with h5py.File(filename,'w') as fw:
            fw['images'] = self.images
            fw['features'] = self.features
            fw['max_images'] = self.max_images
            fw['thumbnail_size'] = self.thumbnail_size
            fw['distance_matrix'] = np.triu(self.distance_matrix)

    def load(self, filename):
        '''
        Clear current object and load variables from HDF5 file.
        :param filename: name of HDF5 file
        :return: nothing
        '''
        with h5py.File(filename,'r') as fr:
            # load as list instead of numpy array
            self.images = []
            for img in fr['images']:
                 self.images.append(img)

            self.features = []
            for feat in fr['features']:
                self.features.append(feat)

            # load whole matrix
            self.distance_matrix = None
            self.distance_matrix = fr['distance_matrix'][:]

            # load as list
            thumbnail_size = fr['thumbnail_size']
            self.thumbnail_size = (thumbnail_size[0], thumbnail_size[1], thumbnail_size[2])

            # reading scalar dataset
            data = fr['max_images']
            self.max_images = data[()]