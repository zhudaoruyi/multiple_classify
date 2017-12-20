import cv2
import time
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from keras.models import load_model

test_dir = '/home/abc/pzw/data/test/'
test_dir_set = glob(test_dir+'*')
print('test sets folders:', test_dir_set, '\n')

for t_dir in test_dir_set:
    img_set = glob(t_dir + '/*')
    print(t_dir,'--images numbers:',len(img_set),'\n')

labels = ['0/', '1/', '20/', '30/', '40/', '50/', '60/', '70/']
classes=['UNKNOWN', 'LINE_CHART', 'AREA_CHART','BAR_CHART','COLUMN_CHART','PIE_CHART','GRID_TABLE','LINE_TABLE']

model = load_model('../model20032100_01_p.h5')

line_column_set = glob(test_dir_set[4] + '/*')
print('sum of images to be predict:', len(line_column_set))


def multi_label_predict(img_dirs):
    y_pred = []
    y_prob = []

    for i, key in tqdm(enumerate(sorted(img_dirs))):
        img = cv2.imread(key)[:,:,::-1]
        img_arr = cv2.resize(img, (299, 299)) / 255.
        y_pred0 = model.predict(np.expand_dims(img_arr, axis=0))
        y_prob.append(y_pred0[0])
    return y_prob

start = time.clock()

y_lc_prob = multi_label_predict(line_column_set)

print('excute time:%s'%(time.clock()-start))


# 存储数据
#with open('Line_column_pred_list.json', 'w') as f:
#    json.dump(y_lc_prob, f)

np.savetxt('lc_prob_round_int.txt', (np.round(y_lc_prob)).astype(np.uint8))

