import os
import cv2
import random
from glob import glob

def mix():
    files = glob(os.path.join('/data/figures/pencil1', '*.jpg'))
    data_size = len(files)
    i = 0
    for file in files:
        index = random.randint(1, 4)
        current_filename = file.replace('pencil1', 'pencil{}'.format(index))
        img = cv2.imread(current_filename)
        cv2.imwrite(file.replace('pencil1', 'pencil'), img)
        i += 1
        print('process: [{} / {}]'.format(i, data_size))

mix()