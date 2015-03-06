import unittest
import numpy as np
import h5py

import img_search.distance_matrix


h5_imgs_fn = '/Users/martin.majer/PycharmProjects/PR4/data/sun_sample.hdf5'
h5_fts_fn = h5_imgs_fn + '.features.hdf5'

with h5py.File(h5_fts_fn,'r') as fr_features, h5py.File(h5_src_fn,'r') as fr_imgs:
    features = np.copy(fr_features['score'])


class DistanceMatrixTestClass(unittest.TestCase):

    def test_distance_matrix_init(self):

        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

        self.assertEqual(dm.max_images, 100000)
        self.assertEqual(dm.thumbnail_size, (150,150,3))
        self.assertEqual(dm.distance_matrix, None)
        self.assertEqual(dm.images, [])
        self.assertEqual(dm.features, [])

    def test_distance_matrix_add_images(self):

        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

        dm.add_images([1,2,3], features[:3,:])
        self.assertEqual(dm.distance_matrix.shape, (3,3))

        dm.add_images([4], features[3,:])
        self.assertEqual(dm.distance_matrix.shape, (4,4))

        dm.add_images([5,6], features[4:6,:])
        self.assertEqual(dm.distance_matrix.shape, (6,6))

if __name__ == '__main__':
    unittest.main()