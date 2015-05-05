__author__ = 'martin.majer'

import sys
import ast
import h5py

indexes = ast.literal_eval(sys.argv[1])
category = sys.argv[2]

print indexes
print category

storage = '/storage/plzen1/home/mmajer/pr4/data/'
filename = storage + 'sun_img_names.hdf5'

with h5py.File(filename, 'r') as fr:
    paths = fr['path']

print paths[:5]



