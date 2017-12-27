from sklearn.model_selection import train_test_split
from os.path import join
from augment import *
from glob import glob
import numpy as np
import threading
import cv2

base_dir = '/home/abc/pzw/data/ten_classes/'

labels_dir = 'labels/'
images_dir = 'images/'

labels = ['0/', '1/', '20/', '30/', '40/', '50/', '60/', '70/']
classes = ['UNKNOWN', 'LINE_CHART', 'AREA_CHART', 'BAR_CHART',
           'COLUMN_CHART', 'PIE_CHART', 'GRID_TABLE', 'LINE_TABLE']


def get_label(label_list, classes=classes):
    """
    label_list:['LINE_CHART', 'COLUMN_CHART']
    return:
        [0,1,0,0,1,0,0,0]
    """
    y = np.zeros(8)
    for m, key in enumerate(label_list):
        if len(key):
            y[classes.index(key)] = 1
    return y


img_dirs = sorted(glob('/home/abc/pzw/data/ten_classes/images/*'))

train_dirs, valid_dirs = train_test_split(img_dirs, test_size=0.2, random_state=42)
#print(len(train_dirs), len(valid_dirs))

class ThreadsafeIter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    def g(*args, **kw):
        return ThreadsafeIter(f(*args, **kw))
    return g


@threadsafe_generator
def data_generator(data_dirs, width, height, batch_size):
    """
    input:
        directories of train or validation,eg:train_dirs,valid_dirs
    output:
        yield X,y
    """

    while True:
        for start in range(0, len(data_dirs), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(data_dirs))
            dirs_batch = data_dirs[start:end]
#            print(dirs_batch)
            for i, key in enumerate(dirs_batch):
#                print(key)
                img = cv2.imread(key)[:, :, ::-1]
                img = cv2.resize(img, (width, height))
                if data_dirs == train_dirs:
                    img = randomCrop(img)
                    img = randomHueSaturationValue(img,
                                                   hue_shift_limit=(-50, 50),
                                                   sat_shift_limit=(-5, 5),
                                                   val_shift_limit=(-15, 15))
                    img = randomShiftScaleRotate(img,
                                                 shift_limit=(-0.0001, 0.0001),
                                                 scale_limit=(-0.1, 0.1),
                                                 rotate_limit=(-10, 10))
                    img = randomHorizontalFlip(img)
                x_batch.append(img)

                with open(join('/home/abc/pzw/data/ten_classes/labels', key.split('/')[-1] + '.txt'), 'r') as f:
                    img_lab = f.read()
                y = get_label(img_lab.split('\n'))
                y_batch.append(y)
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch)
            yield x_batch, y_batch

