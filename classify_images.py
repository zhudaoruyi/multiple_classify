#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import cv2
import shutil
import datetime
import numpy as np
from tqdm import tqdm
from keras.models import load_model

# 9个单类
classes = ['LINE_CHART', 'AREA_CHART', 'BAR_CHART', 'COLUMN_CHART',
           'PIE_CHART', 'UNKNOWN', 'GRID_TABLE', 'LINE_TABLE', 'QR_CODE']
chart_type_dic = {
    0: 'LINE_CHART',
    1: 'AREA_CHART',
    2: 'BAR_CHART',
    3: 'COLUMN_CHART',
    4: 'PIE_CHART',
    5: 'UNKNOWN',
    6: 'GRID_TABLE',
    7: 'LINE_TABLE',
    8: 'QR_CODE'
}


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def list_images(path, file_type='images'):
    """
    列出文件夹中所有的文件，返回
    :param file_type: 'images' or 'any'
    :param path: a directory path, like '../data/pics'
    :return: all the images in the directory
    """
    image_type = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif']
    paths = []
    for file_and_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_and_dir)):
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in image_type:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
            elif file_type == 'any':
                paths.append(os.path.abspath(os.path.join(path, file_and_dir)))
            else:
                if os.path.splitext(file_and_dir)[1] == file_type:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
    return paths


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


# def classify(image):
def classify(pred):
    # img = cv2.imread(image)[:, :, ::-1]
    # img_arr = cv2.resize(img, (299, 299)) / 255.
    # predictions = model.predict(np.expand_dims(img_arr, axis=0))
    predictions = pred
    # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    top_k = predictions[0].argsort()[::-1]
    # 如果最高的是LINE_CHART
    if top_k[0] == 0:
        predictions[0][0] = 1
        predictions[0][2] = predictions[0][4] = predictions[0][5] = predictions[0][6] = predictions[0][7] = 0

    # 如果最高的是AREA_CHART
    elif top_k[0] == 1:
        predictions[0][1] = 1
        predictions[0][2] = predictions[0][4] = predictions[0][5] = predictions[0][6] = predictions[0][7] = 0

    # 如果最高的是BAR_CHART
    elif top_k[0] == 2:
        predictions[0][2] = 1
        predictions[0][0] = predictions[0][1] = predictions[0][3] = predictions[0][4] = predictions[0][5] = \
            predictions[0][6] = predictions[0][7] = 0

    # 如果最高的是COLUMN_CHART
    elif top_k[0]==3:
        predictions[0][3]=1
        predictions[0][2]=predictions[0][4]=predictions[0][5]=predictions[0][6]=predictions[0][7]=0

    # 如果最高的是PIE_CHART
    elif top_k[0]==4:
        predictions[0][4]=1
        predictions[0][0]=predictions[0][1]=predictions[0][2]=predictions[0][3]=predictions[0][5]=\
            predictions[0][6]=predictions[0][7]=0

    # 如果最高的是UNKNOWN
    elif top_k[0]==5:
        predictions[0][5]=1
        predictions[0][0]=predictions[0][1]=predictions[0][2]=predictions[0][3]=predictions[0][4]= \
            predictions[0][6]=predictions[0][7]=0

    # 如果最高的是GRID_TABLE
    elif top_k[0]==6:
        predictions[0][6]=1
        predictions[0][0]=predictions[0][1]=predictions[0][2]=predictions[0][3]=predictions[0][4]=\
            predictions[0][5]=predictions[0][7]=0

    # 如果最高的是LINE_TABLE
    elif top_k[0]==7:
        predictions[0][7]=1
        predictions[0][0]=predictions[0][1]=predictions[0][2]=predictions[0][3]=predictions[0][4]=\
            predictions[0][5]=predictions[0][6]=0

    result = []
    for node_id in top_k:
        human_string = classes[node_id]
        score = predictions[0][node_id]
        # print('%s (score = %.5f)' % (human_string, score))
        if score > 0.8:
            result.append(chart_type_dic[node_id])
    if len(result) == 0:
        # 所有类型的概率都很低, 那就选概率最高的
        node_id = top_k[0]
        if predictions[0][node_id] > 0.5:
            result.append(chart_type_dic[node_id])
        else:
            result.append('UNKNOWN')
    return result


def del_unopen(test_set):
    print '删除打不开的图片...'
    for i, k in tqdm(enumerate(test_set)):
        if cv2.imread(k) is None:
            del (test_set[i])
            os.remove(k)
    print '现在需要分类的图片总数是：%d' % len(test_set)


def predict(test_set, model, batch_size=64):
    x = []
    for i, k in tqdm(enumerate(test_set)):
        img = cv2.imread(k)[:, :, ::-1]
        img = cv2.resize(img, (299, 299))
        x.append(np.array(img)/255.)
    x = np.array(x)
    print 'shape of x:', x.shape
    print 'starting predicting...'
    print 'start time is :', current_time()
    predictions = model.predict(x, batch_size=batch_size)
    print 'end time is :', current_time()
    return predictions


def predict_on_batch(test_set, model, batch_size=64):
    predictions = np.zeros((len(test_set), 9))
    print 'start time is :', current_time()
    for start in tqdm(range(0, len(test_set), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(test_set))
        new_batch = test_set[start:end]
        for img_dir in new_batch:
            img = cv2.imread(img_dir)[:, :, ::-1]
            img = cv2.resize(img, (299, 299))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255.
        batch_pred = model.predict_on_batch(x_batch)
        predictions[start:end] = batch_pred
    print 'end time is :', current_time()
    return np.array(predictions)


def move_image_to_class(test_set, predictions, dst_dirs = 'results/lf_classify_da011001/'):
    # 统计一下数据，低概率的图片，单类的图片，双类的图片，三类的图片
    low_prob_sum, single_sum, double_sum, triple_sum = 0, 0, 0, 0

    print 'start time is :', current_time()
    for ids, pred in enumerate(predictions):
        pred_label = classify([pred])
        pred_label = sorted(pred_label, reverse=True)

        # dst_dirs = 'results/' + 'lf_classify_010802/'
        # dst_dirs = '/home/zhwpeng/data/images0109/' + 'wechat_results/'
        if not os.path.exists(dst_dirs):
            os.mkdir(dst_dirs)

        # 低概率图片
        if pred_label == []:
            if not os.path.exists(dst_dirs + 'low_prob'):
                os.mkdir(dst_dirs + 'low_prob/')
            shutil.copy(test_set[ids], dst_dirs + 'low_prob/')
            low_prob_sum += 1
        # 单类别的图片
        if pred_label != [] and len(pred_label) == 1:
            if not os.path.exists(dst_dirs + pred_label[0]):
                os.mkdir(dst_dirs + pred_label[0])
            shutil.copy(test_set[ids], dst_dirs + pred_label[0])
            single_sum += 1
        # 双类别的图片
        if pred_label != [] and len(pred_label) == 2:
            if not os.path.exists(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1])):
                os.mkdir(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1]))
            shutil.copy(test_set[ids], dst_dirs + str(pred_label[0] + '_and_' + pred_label[1]))
            double_sum += 1
        # 三类别的图片
        if pred_label != [] and len(pred_label) == 3:
            if not os.path.exists(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1] + '_and_' + pred_label[2])):
                os.mkdir(dst_dirs + str(pred_label[0] + '_and_' + pred_label[1] + '_and_' + pred_label[2]))
            shutil.copy(test_set[ids], str(pred_label[0] + '_and_' + pred_label[1] + '_and_' + pred_label[2]))
            triple_sum += 1
    print '低概率图片数量是：%d' % low_prob_sum, '单类图片的数量是：%d' % single_sum, \
        '双类图片的数量是：%d' % double_sum, '三类别图片的数量是：%d' % triple_sum
    print 'end time is :', current_time()
    print 'classify successfully!'
    print '-'*30, '\n', '-'*30


if __name__ == '__main__':
    # 导入模型
    model = load_model('/home/zhwpeng/abc/evaluate_model/models/models0112/m15016_x_01_l.h5', compile=False)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    # 需要分类的数据集
    test_dir = 'pickouts/results011002'
    test_set = sorted(list_images(test_dir))
    print '需要分类的图片总数是：%d' % len(test_set)

    # 预测图片集的类别
    # predictions = predict(test_set, model, batch_size=4)
    predictions = predict_on_batch(test_set, model, batch_size=32)

    # 移动图片到相应的类别文件夹
    move_image_to_class(test_set, predictions, dst_dirs='results/lf_classify_da011003/')
