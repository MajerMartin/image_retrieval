from flask import Flask, request, render_template, send_from_directory
from img_search import kdtree
import os.path
import cv2
import urllib
import numpy as np

# initialize flask server
app = Flask(__name__)

# switch between deployment (True) and local testing mode (False)
caffe_toggle = False

if caffe_toggle:
    import caffe

    # paths for caffe
    caffe_root = '/storage/plzen1/home/campr/metacentrum/caffe/caffe_cemi_7_dev'
    caffe_mean_fn = os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    caffe_deploy_fn = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
    caffe_model_fn = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

    # initialize caffe neural network
    net = caffe.Classifier(caffe_deploy_fn,
                       caffe_model_fn,
                       mean=np.load(caffe_mean_fn),
                       # imagenet uses BGR, same as opencv, swap is not needed:
                       #channel_swap=(2,1,0),
                       # bvlc_reference_caffenet uses pixel values [0, 255], but we don't normalize loaded image to [0.0-1.0], so it is scaled to [0.0, 255.0] already after conversion to float32
                       #raw_scale=255,
                       )
    # run net on cpu
    net.set_mode_cpu()

    # path to data folder
    app.config['UPLOAD_FOLDER'] = '/storage/plzen1/home/mmajer/pr4/data/image_search/'
else:
    import h5py
    from random import randint

    # path to h5df file with features
    h5_fts_fn = '/Users/martin.majer/PycharmProjects/PR4/data/sun_sample.hdf5.features.hdf5'

    # path to data folder
    app.config['UPLOAD_FOLDER'] = '/Users/martin.majer/PycharmProjects/PR4/data/kdt_server/'

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def check_allowed(filename):
    '''
    Check whether uploaded file is image or not.
    :param filename: name of the uploaded image
    :return: boolean value
    '''
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
@app.route('/search')
def search():
    '''
    Main page with form.
    :return: rendered template from html file
    '''
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
    '''
    Add image to database, calculates features and finds nearest neighbors.
    :return: rendered template from html file
    '''

    # get data from form
    search_file = request.files['search_file']
    search_url = request.form['search_url']

    # path to temporary image file
    filename = 'img.jpg'
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # list of images to be passed to render_template
    filenames = []

    ####################################################################################
    # DODELAT AZ BUDE FUNGOVAT HLEDANI PRO UPLOADOVANY OBRAZEK
    if search_url:
        msg = 'Searching using URL (%s)' % search_url
        # pouzit check_allowed
        urllib.urlretrieve(search_url, path)
    ####################################################################################

    elif search_file and check_allowed(search_file.filename):
        msg = 'Searching using file (filename: %s)' % search_file.filename

        # docasne ulozit obrazek na disk
        search_file.save(path)

        # otevrit obrazek
        img = cv2.imread(path)
        print 'img.shape\t|\t', img.shape
        # ukladat i original?
        # obratit barvy
        # oriznout na ctverec pro zachovani pomeru stran

        # spocitat features - zatim random
        with h5py.File(h5_fts_fn, 'r') as fr_features:
            img_feat = fr_features['score'][randint(0, 9999)]
            print 'img_feat.shape\t|\t', img_feat.shape

        # zavolat add_images()
        print 'len(kdt.features) before\t|\t', len(kdt.features)
        kdt.add_images([img], [img_feat])
        print 'len(kdt.features) after\t|\t', len(kdt.features)

        # zavolat save()
        kdt.save()

        # smazat obrazek
        os.remove(path)

        # zavolat find()
        indexes, distances = kdt.find_k_nearest_by_index(len(kdt.features) - 1, 6)

        filenames = []
        for index in indexes:
            filenames.append(str(index) + '.jpg')

        print filenames

    else:
        msg = 'Please provide URL or file.'

    return render_template('results.html', msg=msg, filenames=filenames)

@app.route('/thumbs/<filename>')
def send_file(filename):
    '''
    Send image from directory to server.
    :param filename: name of the image
    :return: image
    '''
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'thumbs'), filename)

if __name__ == '__main__':
    kdt = kdtree.ImageSearchKDTree(app.config['UPLOAD_FOLDER'], n, (150,150,3))
    app.run(port=8080, debug=True)