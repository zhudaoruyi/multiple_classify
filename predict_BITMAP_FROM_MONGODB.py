import re
import os
import cv2
import time
import json
import numpy as np
from tqdm import tqdm
from keras.models import load_model

# 八个单类
labels = ['0/', '1/', '20/', '30/', '40/', '50/', '60/', '70/']
classes = ['UNKNOWN', 'LINE_CHART', 'AREA_CHART', 'BAR_CHART',
           'COLUMN_CHART', 'PIE_CHART', 'GRID_TABLE', 'LINE_TABLE']

# 导入模型
model = load_model('model20032100_01_p.h5')

# 被测文件夹
test_dir = '/home/zhwpeng/data/BITMAP_FROM_MONGODB/'
# 被测文件夹下的文件夹名
obj_dirs = ['area', 'bar', 'column', 'column_3D', 'Discrete+line', 'line', 'line+area', 'line+bar',
            'line_special1', 'line_special2', 'Multi', 'other', 'pie', 'Radar_chart', 'table']


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


# 与glob类似功能的函数，只是对图片格式有效
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


start = time.clock()

for ids, direcs in enumerate(obj_dirs):
    # test_set = sorted(glob(test_dir + obj_dirs[4] + '/*.jpg'))
    test_set = sorted(list_pictures(test_dir + direcs))
    print('sum of images in %s directory to be predict: ' % direcs, len(test_set))
    if len(test_set):
        _, pred_labels = multi_label_predict(test_set)

        print('executing time:%s' % (time.clock()-start))
        print('prediction finished!!!')
        print('-'*60, '\n', '-'*60)

        dst_dir = 'BITMAP_FROM_MONGODB_json/'
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        # 存储数据
        # with open(dst_dir + 'test_set_dirs.json', 'w') as fi:
        #     json.dump(test_set, fi)

        # with open(dst_dir + 'pred_labels.json', 'w') as f:
        #     json.dump(pred_labels, f)

        test_set_prefix = [test_set[i].split('/')[-1] for i in range(len(test_set))]

        # 图片名和对应的类别存入到字典里，保存json
        with open(dst_dir + direcs + '.json', 'w') as fil:
            json.dump(dict(zip(test_set_prefix, pred_labels)), fil)

        # np.savetxt(dst_dir + 'pred.txt', y_pred)
        print('executing time:%s'%(time.clock()-start))
        print('saved successfully!')
        print('-'*60, '\n', '-'*60)

