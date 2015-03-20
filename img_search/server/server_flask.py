from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

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
    elif search_file:
        msg = 'Searching using file (name %s)' % (search_file.filename)
    else:
        msg = 'Please provide URL or file.'

    return render_template('results.html', msg=msg)



if __name__ == '__main__':
    # zde bude inicializace vyhledavace, pak se spusti server...

    app.run(port=8080, debug=True)
