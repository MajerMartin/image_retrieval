__author__ = 'martin.majer'

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
import h5py
import shutil

import img_search.kdtree_flann


class KDTreeFlannTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h5_imgs_fn = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sun_sample.hdf5')
        h5_fts_fn = h5_imgs_fn + '.features.hdf5'

        with h5py.File(h5_fts_fn, 'r') as fr_features, h5py.File(h5_imgs_fn, 'r') as fr_imgs:
            cls.features = np.copy(fr_features['score'])
            imgs = np.array(fr_imgs['imgs'][:7][:, :, ::-1])
            cls.imgs = imgs.astype(np.float32) * 255

        cls.storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_flann_test')

    @classmethod
    def tearDown(cls):
        shutil.rmtree(cls.storage_dir)

    def test_kdtree_flann_init(self):
        kdt = img_search.kdtree_flann.ImageSearchKDTreeFlann(self.storage_dir)

        self.assertEqual(kdt.max_images, 100000)
        self.assertEqual(kdt.thumbnail_size, (150, 150, 3))
        self.assertEqual(kdt.tree, 'kdtree')
        self.assertEqual(kdt.features, [])
        self.assertEqual(kdt.filename, 'data.hdf5')
        self.assertEqual(kdt.thumbs, 'thumbs')
        self.assertEqual(kdt.data_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_flann_test', 'data.hdf5'))
        self.assertEqual(kdt.tree_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_flann_test', 'kdtree'))
        self.assertEqual(kdt.storage_dir, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_flann_test'))
        self.assertEqual(os.path.isfile(kdt.data_path), True)
        self.assertEqual(os.path.isfile(kdt.tree_path), False)

    def test_kdtree_flann_add_images(self):
        kdt = img_search.kdtree_flann.ImageSearchKDTreeFlann(self.storage_dir)

        kdt.add_images(self.imgs[:3, :, ::-1], self.features[:3, :])
        self.assertNotEqual(kdt.tree, None)
        self.assertEqual(len(kdt.features), 3)

        kdt.add_images([self.imgs[3, :, ::-1]], [self.features[3, :]])
        self.assertNotEqual(kdt.tree, None)
        self.assertEqual(len(kdt.features), 4)

        kdt.add_images(self.imgs[4:6, :, ::-1], self.features[4:6, :])
        self.assertNotEqual(kdt.tree, None)
        self.assertEqual(len(kdt.features), 6)

    def test_kdtree_flann_get_images(self):
        kdt = img_search.kdtree_flann.ImageSearchKDTreeFlann(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])

        images = kdt.get_images([1, 3, 5])
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0].shape, (150, 150, 3))

        #for img in images:
        #    plt.figure()
        #    plt.imshow(img[:,:,::-1])
        #    plt.show()

        images = kdt.get_images([1])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (150, 150, 3))

    def test_kdtree_flann_find_k_nearest_by_index(self):
        kdt = img_search.kdtree_flann.ImageSearchKDTreeFlann(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])

        indexes, distances = kdt.find_k_nearest_by_index(1)
        self.assertEqual(len(indexes), 3)
        self.assertEqual(len(distances), 3)

    def test_kdtree_flann_remove_last_image(self):
        kdt = img_search.kdtree.ImageSearchKDTree(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])

        last = len(kdt.features) - 1
        self.assertEqual(len(kdt.features), 6)
        assert os.path.exists(os.path.join(self.storage_dir, 'thumbs', str(last) + '.jpg')) == 1

        kdt.remove_last_image()

        self.assertEqual(len(kdt.features), 5)
        assert os.path.exists(os.path.join(self.storage_dir, 'thumbs', str(last) + '.jpg')) == 0

    def test_kdtree_flann_save(self):
        kdt = img_search.kdtree_flann.ImageSearchKDTreeFlann(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])
        kdt.save()

        self.assertEqual(os.path.isfile(kdt.data_path), True)
        self.assertEqual(os.path.isfile(kdt.tree_path), True)

    def test_kdtree_flann_load(self):
        kdt = img_search.kdtree_flann.ImageSearchKDTreeFlann(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])
        kdt.save()

        kdt_copy = img_search.kdtree_flann.ImageSearchKDTreeFlann(self.storage_dir)

        self.assertEqual(kdt_copy.max_images, 100000)
        self.assertEqual(kdt_copy.thumbnail_size, (150, 150, 3))
        self.assertEqual(kdt_copy.tree, 'kdtree')
        self.assertEqual(len(kdt_copy.features), 6)
        self.assertEqual(kdt_copy.features[0].shape, (1000,))
        self.assertEqual(kdt_copy.filename, 'data.hdf5')
        self.assertEqual(kdt_copy.thumbs, 'thumbs')
        self.assertEqual(kdt_copy.data_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_flann_test', 'data.hdf5'))
        self.assertEqual(kdt_copy.tree_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_flann_test', 'kdtree'))
        self.assertEqual(kdt_copy.storage_dir, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_flann_test'))
        self.assertEqual(os.path.isfile(kdt_copy.data_path), True)
        self.assertEqual(os.path.isfile(kdt_copy.tree_path), True)

        indexes, distances = kdt.find_k_nearest_by_index(1)
        indexes_copy, distances_copy = kdt_copy.find_k_nearest_by_index(1)

        self.assertEqual(indexes, indexes_copy)
        self.assertEqual(distances, distances_copy)


if __name__ == '__main__':
    unittest.main()