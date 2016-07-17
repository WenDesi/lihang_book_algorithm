#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))

    return features

def Predict(testset,trainset,train_labels):
    predict = []
    count = 0

    for test_vec in testset:
        # 输出当前运行的测试用例坐标，用于测试
        print count
        count += 1

        knn_list = []       # 当前k个最近邻居
        max_index = -1      # 当前k个最近邻居中距离最远点的坐标
        max_dist = 0        # 当前k个最近邻居中距离最远点的距离

        # 先将前k个点放入k个最近邻居中，填充满knn_list
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]

            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离

            knn_list.append((dist,label))

        # 剩下的点
        for i in range(k,len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]

            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离

            # 寻找10个邻近点钟距离最远的点
            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]

            # 如果当前k个最近邻居中存在点距离比当前点距离远，则替换
            if dist < max_dist:
                knn_list[max_index] = (dist,label)
                max_index = -1
                max_dist = 0


        # 统计选票
        class_total = 10
        class_count = [0 for i in range(class_total)]
        for dist,label in knn_list:
            class_count[label] += 1

        # 找出最大选票
        mmax = max(class_count)

        # 找出最大选票标签
        for i in range(class_total):
            if mmax == class_count[i]:
                predict.append(i)
                break

    return np.array(predict)

k = 10

if __name__ == '__main__':

    print 'Start read data'

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    features = get_hog_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print 'read data cost ',time_2 - time_1,' second','\n'

    print 'Start training'
    print 'knn do not need to train'
    time_3 = time.time()
    print 'training cost ',time_3 - time_2,' second','\n'

    print 'Start predicting'
    test_predict = Predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print 'predicting cost ',time_4 - time_3,' second','\n'

    score = accuracy_score(test_labels,test_predict)
    print "The accruacy socre is ", score