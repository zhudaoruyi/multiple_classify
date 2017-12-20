import numpy as np

# with open('l_c_prob.txt', 'r') as  f:
#     data = f.read()
#
# # print(data, type(data))
#
# print(data[0])
#
# a = np.random.random(10)
# print(a)
#
# b = np.round(a)
# print(b)

# labels = ['0/', '1/', '20/', '30/', '40/', '50/', '60/', '70/']
# classes=['UNKNOWN', 'LINE_CHART', 'AREA_CHART', 'BAR_CHART', 'COLUMN_CHART', 'PIE_CHART', 'GRID_TABLE', 'LINE_TABLE']

# prob = np.random.random((2,8))
# print(prob)
#
# prob_round = np.round(prob)
# print(prob_round)
#
# index = np.nonzero(prob_round[0])
# print(index)
# print(index[0])
#
# pred_label = []
# for i in index[0]:
#     print(classes[i])
#     pred_label.append(classes[i])
#
# print(pred_label)
#
#
# def get_label_from_pred(pred, classes=classes):
#     """
#     :param pred:should be 1D array or list,like [0,0,0,1,0,0,1,0]
#     :param classes: default classes=['UNKNOWN', 'LINE_CHART', 'AREA_CHART', 'BAR_CHART',
#                                     'COLUMN_CHART', 'PIE_CHART', 'GRID_TABLE', 'LINE_TABLE']
#     :return:
#     """
#     pred_label = []
#     index = np.nonzero(np.round(pred))[0]
#     for i in index:
#         pred_label.append(classes[i])
#     return pred_label

# 归类到对应的文件夹里
from tqdm import tqdm
from glob import glob
import shutil
import json
import time
import os

start = time.clock()

# test_dir = '/home/abc/pzw/data/test/'
# test_dir_set = glob(test_dir+'*')
# line_column_set = sorted(glob(test_dir_set[4] + '/*'))

line_column_set = sorted(glob('/home/zhwpeng/data/test/alltests/*'))

with open("prediction_results/lc_pred_labels.json", "r", encoding='utf-8') as f:
    lc_pred_labels = json.loads(f.read())
print(len(lc_pred_labels), '\n', type(lc_pred_labels))

dst_dirs = 'multi_classify1/'
if not os.path.exists(dst_dirs):
    os.mkdir(dst_dirs)

# 统计一下数据，低概率的图片，单类的图片，双类的图片，三类的图片
low_prob_sum, single_sum, double_sum, triple_sum = 0, 0, 0, 0

for i, key in tqdm(enumerate(lc_pred_labels)):
    # 低概率图片
    if key == []:
        if not os.path.exists(dst_dirs + 'low_prob'):
            os.mkdir(dst_dirs + 'low_prob/')
        shutil.copy(line_column_set[i], dst_dirs + 'low_prob/')
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
