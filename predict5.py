import os
import cv2
import time
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from keras.models import load_model

test_dir = '/home/abc/hdd/bitmap_dataset/'

obj_dirs = ['BAR_CHART', 'LINE_BAR_CHART', 'LINE_CHART', 'PIE_CHART']

test_dir_set = glob(test_dir+'*')
print('test sets folders:', test_dir_set, '\n')

for t_dir in test_dir_set:
    if t_dir in obj_dirs:
        img_set = glob(t_dir + '/*')
        print(t_dir,'--images numbers:',len(img_set),'\n')

labels = ['0/', '1/', '20/', '30/', '40/', '50/', '60/', '70/']
classes=['UNKNOWN', 'LINE_CHART', 'AREA_CHART','BAR_CHART','COLUMN_CHART','PIE_CHART','GRID_TABLE','LINE_TABLE']

model = load_model('../model20032100_01_p.h5')

#line_column_set = glob(test_dir_set[4] + '/*')
try:
    test_set = sorted(glob(test_dir + obj_dirs[0] + '/*'))
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

dst_dir = 'prediction_results_of_bitmap_dataset/'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

 dict(zip(test_set, pred_labels))

# 存储数据
with open(dst_dir + 'test_set_dirs.json', 'w') as fi:
    json.dump(test_set, fi)

with open(dst_dir + 'pred_labels.json', 'w') as f:
    json.dump(pred_labels, f)

# 图片名和对应的类别存入到字典里，保存json
with open(dst_dir + 'BAR.json', 'w') as fil:
    json.dump(dict(zip(test_set, pred_labels)), fil)

np.savetxt(dst_dir + 'pred.txt', y_pred)
print('executing time:%s'%(time.clock()-start))
print('saved successfully!')
print('-'*60, '\n', '-'*60)

'''
print('*****start classify images and copy to dst directory!!!*****')
print('-'*60, '\n', '-'*60)

save_dirs = obj_dirs[0] + '_classify/'
if not os.path.exists(save_dirs):
    os.mkdir(save_dirs)

# 统计一下数据，低概率的图片，单类的图片，双类的图片，三类的图片
low_prob_sum, single_sum, double_sum, triple_sum = 0, 0, 0, 0

for i, key in tqdm(enumerate(pred_labels)):
    # 低概率图片
    if key == []:
        if not os.path.exists(save_dirs + 'low_prob'):
            os.mkdir(save_dirs + 'low_prob/')
        shutil.copy(line_column_set[i], save_dirs + 'low_prob/')
        low_prob_sum += 1
    # 单类别的图片
    if key != [] and len(key) == 1:
        if not os.path.exists(dst_dirs + key[0]):
            os.mkdir(dst_dirs + key[0])
        shutil.copy(line_column_set[i], dst_dirs + key[0])
        single_sum += 1
    # 双类别的图片
    if key != [] and len(key) == 2:
        if not os.path.exists(dst_dirs + str(key[0] + '_and_' + key[1])):
            os.mkdir(dst_dirs + str(key[0] + '_and_' + key[1]))
        shutil.copy(line_column_set[i], dst_dirs + str(key[0] + '_and_' + key[1]))
        double_sum += 1
    # 三类别的图片
    if key != [] and len(key) == 3:
        if not os.path.exists(dst_dirs + str(key[0] + '_and_' + key[1] + '_and_' + key[2])):
            os.mkdir(dst_dirs + str(key[0] + '_and_' + key[1] + '_and_' + key[2]))
        shutil.copy(line_column_set[i], str(key[0] + '_and_' + key[1] + '_and_' + key[2]))
        triple_sum += 1
print('total images: %s' % len(line_column_set), '\n', 'single class images: %s' % single_sum,
      '\n', 'double classes images: %s' % double_sum, '\n', 'triple classes images: %s' % triple_sum,
      '\n', 'low probably images: %s' % low_prob_sum)
print('executing time:%s s' % (time.clock() - start))

'''
