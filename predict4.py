import os
import cv2
import time
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from keras.models import load_model

test_dir = '/home/abc/pzw/data/ten_classes/images/'
#test_dir_set = glob(test_dir+'*')
#print('test sets folders:', test_dir_set, '\n')

#for t_dir in test_dir_set:
#    img_set = glob(t_dir + '/*')
#    print(t_dir,'--images numbers:',len(img_set),'\n')

labels = ['0/', '1/', '20/', '30/', '40/', '50/', '60/', '70/']
classes=['UNKNOWN', 'LINE_CHART', 'AREA_CHART','BAR_CHART','COLUMN_CHART','PIE_CHART','GRID_TABLE','LINE_TABLE']

model = load_model('../model20032100_01_p.h5')

#line_column_set = glob(test_dir_set[4] + '/*')
try:
    test_set = sorted(glob(test_dir + '*'))
    print('sum of images to be predict:', len(test_set))
except:
    print('wrong diretory!')


def get_label_from_pred(pred, classes=classes):
    """
    :param pred:should be 1D array or list,like [0,0,0,1,0,0,1,0] 
    :param classes: default classes=['UNKNOWN', 'LINE_CHART', 'AREA_CHART', 'BAR_CHART', 
                                    'COLUMN_CHART', 'PIE_CHART', 'GRID_TABLE', 'LINE_TABLE']
    :return: 
    """
    pred_label = []
    index = np.nonzero(np.round(pred))[0]
    for i in index:
        pred_label.append(classes[i])
    return pred_label


def multi_label_predict(img_dirs):
    y_pred = []
    pred_labels = []
    for i, key in tqdm(enumerate(sorted(img_dirs))):
        img = cv2.imread(key)[:,:,::-1]
        img_arr = cv2.resize(img, (299, 299)) / 255.
        y_pred_single = model.predict(np.expand_dims(img_arr, axis=0))
        y_pred.append(y_pred_single[0])

        pred_label = get_label_from_pred(y_pred_single[0])
        pred_labels.append(pred_label)
    return y_pred, pred_labels

start = time.clock()

y_pred, pred_labels = multi_label_predict(test_set)

print('executing time:%s'%(time.clock()-start))

start = time.clock()

dst_dir = 'prediction_results_of_train/'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

# 存储数据
with open(dst_dir + 'test_set_dirs.json', 'w') as fi:
    json.dump(test_set, fi)

with open(dst_dir + 'pred_labels.json', 'w') as f:
    json.dump(pred_labels, f)

np.savetxt(dst_dir + 'pred.txt', y_pred)
print('executing time:%s'%(time.clock()-start))

