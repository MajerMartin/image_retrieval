from flask import Flask, request, render_template, send_from_directory
from img_search import kdtree
import os.path
import cv2
import h5py
from random import randint

# pak nebude potreba
h5_imgs_fn = '/Users/martin.majer/PycharmProjects/PR4/data/sun_sample.hdf5'
h5_fts_fn = h5_imgs_fn + '.features.hdf5'
# jaky limit?
n = 100000

app = Flask(__name__)

# pravdepodobne zmena na metacentrum
app.config['UPLOAD_FOLDER'] = '/Users/martin.majer/PycharmProjects/PR4/data/kdt_server/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def check_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
@app.route('/search')
def search():
    # uvodni stranka s vyhledavacim formularem
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
    # vyhledavani, bud podle URL s obrazkem nebo podle nahraneho obrazku
    search_file = request.files['search_file']
    search_url = request.form['search_url']

    if search_url:
        msg = 'Searching using URL (%s)' % (search_url)
    elif search_file and check_allowed(search_file.filename):
        msg = 'Searching using file (name %s)' % (search_file.filename)

        # docasne ulozit obrazek na disk
        filename = 'img.jpg'
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        search_file.save(path)

        # otevrit obrazek
        img = cv2.imread(path)
        print 'img.shape\t|\t', img.shape
        # upravit zmenseni KDT pro zachovani ratia ?

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
        indexes = kdt.find_k_nearest_by_index(len(kdt.features) - 1)

        filenames = []
        for index in indexes:
            filenames.append(str(index) + '.jpg')

        print filenames

    else:
        msg = 'Please provide URL or file.'

    return render_template('results.html', msg=msg, filenames=filenames)

@app.route('/thumbs/<filename>')
def send_file(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'thumbs'), filename)

if __name__ == '__main__':
    kdt = kdtree.ImageSearchKDTree(app.config['UPLOAD_FOLDER'], n, (150,150,3))
    app.run(port=8080, debug=True)