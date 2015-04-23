# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Compute Caffe features for images stored in h5 file
# ===================================================

# <codecell>

import os
import h5py
import caffe
import zipfile
import numpy as np
import cv2

try:
    # run from ipython?
    __IPYTHON__

    print 'ipy mode'

    from IPython import get_ipython
    ipy = get_ipython()

    ipy.magic("matplotlib inline")
    ipy.magic("config InlineBackend.figure_format = 'png'")

    # load other ipython resources
    import matplotlib.pyplot as plt

    debug = 50

except NameError:
    # run from python
    ipy = None
    debug = 0

print os.getcwd()

# <codecell>

# options
h5_src_fn = '/storage/plzen3-kky/korpusy/cv/campr-image-search/data/sun_sample_majer_20150222.hdf5'
h5_dst_fn = h5_src_fn + '.features.hdf5'
n = 10000

# caffe
gpu = True
caffe_root = '/storage/plzen1/home/campr/metacentrum/caffe/caffe_cemi_7_dev'
caffe_mean_fn = os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
caffe_deploy_fn = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
caffe_model_fn = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
caffe_labels_fn = os.path.join(caffe_root, 'data/ilsvrc12/synset_words.txt')

# <codecell>

net = caffe.Classifier(caffe_deploy_fn,
                       caffe_model_fn,
                       mean=np.load(caffe_mean_fn),
                       # imagenet uses BGR, same as opencv, swap is not needed:
                       #channel_swap=(2,1,0),
                       # bvlc_reference_caffenet uses pixel values [0, 255], but we don't normalize loaded image to [0.0-1.0], so it is scaled to [0.0, 255.0] already after conversion to float32
                       #raw_scale=255,
                       )
                       
net.set_phase_test()   

if gpu:
    net.set_mode_gpu()
else:
    net.set_mode_cpu()

# <codecell>

if debug:
    labels = np.loadtxt(caffe_labels_fn, str, delimiter='\t')

# <codecell>

print 'net info:'
print
print 'model:', caffe_model_fn
print
print 'deploy:', caffe_deploy_fn
print
print 'net params:'
                       
for p in [(k, v[0].data.shape) for k, v in net.params.items()]:
    print p
                       
print
print 'net blobs:'
                       
for p in [(k, v.data.shape) for k, v in net.blobs.items()]:
    print p

# <codecell>

with h5py.File(h5_src_fn,'r') as fr, h5py.File(h5_dst_fn, 'w') as fw:
    print 'opening h5 file'

    fw.create_dataset('filename', (n, ), dtype=h5py.special_dtype(vlen=unicode))
    fw.create_dataset('blob_fc7', (n, 10, 4096), dtype=np.float32)
    fw.create_dataset('blob_fc8', (n, 10, 1000), dtype=np.float32)
    fw.create_dataset('blob_prob', (n, 10, 1000), dtype=np.float32)
    fw.create_dataset('score', (n, 1000), dtype=np.float32)

    print 'processing images'

    for i in range(0, n):
        if debug and i > debug:
            break

        fn = fr['filenames'][i]
            
        print '\r%6d/%6d' % (i, n), fn,

        img = np.array(fr['imgs'][i][:,:,::-1]) # are channels swapped in source h5 file?
        img = img.astype(np.float32) * 255 # this caffe model expects pixel values from 0 to 255

        assert img.shape == (227, 227, 3)
        
        # predict
        score = net.predict([img]).flatten()  # predict() takes any number of images, and formats them for the Caffe net automatically
        
        # output to h5
        fw['filename'][i] = fn
        fw['score'][i] = score
        fw['blob_prob'][i] = net.blobs['prob'].data.squeeze()
        fw['blob_fc8'][i] = net.blobs['fc8'].data.squeeze()
        fw['blob_fc7'][i] = net.blobs['fc7'].data.squeeze()

        if debug:
            plt.figure()
            plt.imshow(img.astype(np.uint8)[:,:,::-1])
            plt.colorbar()
            plt.show()

            top_k = score.argsort()[-1:-6:-1]
            print labels[top_k]

            print
            print
            
print
print 'done.'        

# <codecell>

# ukoly:
# 1) vypocitat distance matrix pro vsechny obrazky, lze pouzit np.linalg.norm (viz nize). Jako priznaky pouzit "score" a "blob_fc7", u ktereho se z 10ti vektoru vybere ten na indexu 4
# 2) pouzit pro to same http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html, http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html#scipy.spatial.distance.squareform
# 3) vytvorit funkci find_nearest_images(img_idx, k), k = pocet nejblizsich obrazku, ktere se maji najit
# 4) pro rychlejsi vyhledavani zkusit http://scikit-learn.org/stable/modules/neighbors.html#unsupervised-nearest-neighbors

if debug:
    with h5py.File(h5_src_fn,'r') as fr_imgs, h5py.File(h5_dst_fn,'r') as fr_features:
        print fr_features.keys()
        for k,v in fr_features.iteritems():
            print k, v.shape
            
        i = 105
            
        img_1 = np.array(fr_imgs['imgs'][i][:,:,::-1]) # are channels swapped in source h5 file?
        img_1 = img_1.astype(np.float32) * 255 # this caffe model expects pixel values from 0 to 255
        
        plt.figure()
        plt.imshow(img_1.astype(np.uint8)[:,:,::-1])
        plt.show()
        
        best_d = None
        best_j = None
        ds = []
        features_i = np.copy(fr_features['score'][i])
        
        for j in range(0, n):
            features_j = fr_features['score'][j]
            d = np.linalg.norm(features_i - features_j)
            
            ds.append(d)
            
            if best_d is None or best_d > d:
                best_d = d
                best_j = j
                
        ds = np.array(ds)        
            
        best_idxs = np.argsort(ds)
        
        # show 10 best (=similar) images
        for k in range(0, 10):
            j = best_idxs[k]
            print ds[j]
            
            img_2 = np.array(fr_imgs['imgs'][j][:,:,::-1]) # are channels swapped in source h5 file?
            img_2 = img_2.astype(np.float32) * 255 # this caffe model expects pixel values from 0 to 255

            plt.figure()
            plt.imshow(img_2.astype(np.uint8)[:,:,::-1])
            plt.show()            

