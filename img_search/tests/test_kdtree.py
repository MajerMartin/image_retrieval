# __author__ = 'martin.majer'
#
# import unittest
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
#
# import img_search.kdtree
#
#
# h5_imgs_fn = '/Users/martin.majer/PycharmProjects/PR4/data/sun_sample.hdf5'
# h5_fts_fn = h5_imgs_fn + '.features.hdf5'
#
# with h5py.File(h5_fts_fn,'r') as fr_features, h5py.File(h5_imgs_fn,'r') as fr_imgs:
#     features = np.copy(fr_features['score'])
#     imgs = np.array(fr_imgs['imgs'][:7][:,:,::-1])
#     imgs = imgs.astype(np.float32) * 255
#
# storage_dir = '/Users/martin.majer/PycharmProjects/PR4/data/kdt/'
#
# class KDTreeTestClass(unittest.TestCase):
#
#     def test_kdtree_init(self):
#
#         kdt = img_search.kdtree.ImageSearchKDTree(storage_dir)
#
#         self.assertEqual(kdt.max_images, 100000)
#         self.assertEqual(kdt.thumbnail_size, (150,150,3))
#         self.assertEqual(kdt.tree, None)
#         self.assertEqual(kdt.features, [])
#         self.assertEqual(kdt.data_path, '/Users/martin.majer/PycharmProjects/PR4/data/kdt/data_kdt.hdf5')
#
#     def test_kdtree_add_images(self):
#
#         kdt = img_search.kdtree.ImageSearchKDTree(storage_dir)
#
#         kdt.add_images(imgs[:3,:,::-1], features[:3,:])
#         self.assertNotEqual(kdt.tree, None)
#         self.assertEqual(len(kdt.features), 3)
#
#         kdt.add_images([imgs[3,:,::-1]], [features[3,:]])
#         self.assertNotEqual(kdt.tree, None)
#         self.assertEqual(len(kdt.features), 4)
#
#         kdt.add_images(imgs[4:6,:,::-1], features[4:6,:])
#         self.assertNotEqual(kdt.tree, None)
#         self.assertEqual(len(kdt.features), 6)
#
#     def test_kdtree_get_images(self):
#
#         kdt = img_search.kdtree.ImageSearchKDTree(storage_dir)
#
#         kdt.add_images(imgs[:6,:,::-1], features[:6,:])
#
#         images = kdt.get_images([1,3,5])
#         self.assertEqual(len(images), 3)
#         self.assertEqual(images[0].shape, (150, 150, 3))
#
#         #for img in images:
#         #    plt.figure()
#         #    plt.imshow(img[:,:,::-1])
#         #    plt.show()
#
#         images = kdt.get_images([1])
#         self.assertEqual(len(images), 1)
#         self.assertEqual(images[0].shape, (150, 150, 3))
#
#     def test_kdtree_find_k_nearest_by_index(self):
#
#         kdt = img_search.kdtree.ImageSearchKDTree(storage_dir)
#
#         kdt.add_images(imgs[:6,:,::-1], features[:6,:])
#
#         indexes = kdt.find_k_nearest_by_index(1)
#         self.assertEqual(len(indexes), 3)
#
#     def test_kdtree_load(self):
#
#         kdt = img_search.kdtree.ImageSearchKDTree(storage_dir)
#
#         kdt.add_images(imgs[:6,:,::-1], features[:6,:])
#         kdt.save()
#
#         kdt_copy = img_search.kdtree.ImageSearchKDTree(storage_dir)
#
#         self.assertEqual(len(kdt_copy.features), 6)
#         self.assertEqual(kdt_copy.features[0].shape, (1000,))
#         self.assertNotEqual(kdt_copy.tree, None)
#         self.assertEqual(kdt_copy.thumbnail_size, (150,150,3))
#         self.assertEqual(kdt_copy.max_images, 100000)
#         self.assertEqual(kdt_copy.data_path, '/Users/martin.majer/PycharmProjects/PR4/data/kdt/data_kdt.hdf5')
#         self.assertEqual(kdt_copy.storage_dir, '/Users/martin.majer/PycharmProjects/PR4/data/kdt/')
#
# if __name__ == '__main__':
#     unittest.main()