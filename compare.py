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

count = 0

with h5py.File(filename, 'r') as fr:
    for i in indexes:
        str = fr['path'][i]
        print str
        if category in str:
            count += 1

print count