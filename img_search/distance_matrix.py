__author__ = 'martin.majer'

import numpy as np

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
                    print "i:", i, ", j:", j
                    new_col[j,0] = np.linalg.norm(self.features[i] - self.features[j])

                self.distance_matrix = np.concatenate((self.distance_matrix, new_col), axis=1)






        self.images.extend(images)


        print "imgs: ", self.images
        print "fts: ", len(self.features), '\n'



        # zmensi obrazky
        # prida nove obrazky do images
        return self.distance_matrix
        # features - do teto metody vstupuji jiz zvolene typy priznaku
        # images - zmensit dle self.thumbnail_size, fixni velikost




    def find_k_nearest_by_index(self, img_index, k=3):
        pass

    def get_images(self, indexes):
        """

        :param indexes:
        :return:
        """
        images = []
        for index in indexes:
            images.append(self.images[index])
        return images

    def save(self, filename):
        # ulozit do H5, klice images, features, max_images, thumbnail_size, distance_matrix (ulozit jen spodni trojuhelnik)
        pass

    def load(self, filename):
        pass