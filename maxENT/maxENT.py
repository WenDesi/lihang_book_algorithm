# encoding=utf-8
# @Author: WenDesi
# @Date:   05-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 06-11-16

import cv2
import csv
import math
import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class MaxEnt(object):

    def init_params(self, X, Y):
        self.X_ = X
        self.Y_ = set()

        self.cal_Pxy_Px(X, Y)

        self.N = len(X)
        self.n = len(self.Pxy)
        self.M = 2.0

        self.build_dict()
        self.cal_EPxy()

    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}

        for i, (x, y) in enumerate(self.Pxy):
            self.id2xy[i] = (x, y)
            self.xy2id[(x, y)] = i

    def cal_Pxy_Px(self, X, Y):
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)

        for i in xrange(len(X)):
            x_, y = X[i], Y[i]
            self.Y_.add(y)

            for x in x_:
                self.Pxy[(x, y)] += 1
                self.Px[x] += 1

    def cal_EPxy(self):
        self.EPxy = defaultdict(float)
        for id in xrange(self.n):
            (x, y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x, y)]) / float(self.N)

    def cal_pyx(self,X,y):
        result = 0.0
        for x in X:
            if self.fxy(x, y):
                id = self.xy2id[(x, y)]
                result += self.w[id]
        print result
        return (math.exp(result),y)

    def cal_probality(self, X):
        Pyxs = [(self.cal_pyx(X,y)) for y in self.Y_]
        Z = sum([prob for prob, y in Pyxs])
        return [(prob/Z,y) for prob,y in Pyxs]


    def cal_EPx(self):
        self.EPx = [0.0 for i in xrange(self.n)]

        for i,X in enumerate(self.X_):
            Pyxs = self.cal_probality(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x, y)]

                        self.EPx[id] += Pyx * (1.0 / self.N)

    def fxy(self, x, y):
        return (x, y) in self.xy2id

    def train(self, X, Y):
        self.init_params(X, Y)
        self.w = [0.0 for i in range(self.n)]

        max_iteration = 1000
        for times in xrange(max_iteration):
            print 'iterater times %d' % times
            sigmas = []
            self.cal_EPx()
            print self.EPx[243]

            for i in xrange(self.n):
                try:
                    sigma = 1 / self.M * math.log(self.EPxy[i] / self.EPx[i])
                except:
                    print i
                    print self.EPx[i]
                    print self.id2xy[i]
                    return
                sigmas.append(sigma)

            if len(filter(lambda x: abs(x) >= 0.01, sigmas)) == 0:
                break

            self.w = [self.w[i] + sigmas[i] for i in xrange(self.n)]

    def predict(self, testset):
        results = []
        for test in testset:
            result = self.cal_probality(test)
            results.append(max(result, key=lambda x: x[0])[1])
        return results


def rebuild_X(X):
    new_X = []
    for x in X:
        new_x = []
        for i in range(len(x)):
            new_x.append(i * 2 + x[i])
        new_X.append(new_x)

    print 'end build'
    return new_X

#
# def read_demo(filepath='C:/Users/1501213972/Desktop/data.txt'):
#     features = []
#     labels = []
#     for line in open(filepath, "r"):
#         sample = line.strip().split("\t")
#         if len(sample) < 2:  # 至少：标签+一个特征
#             continue
#         y = sample[0]
#         X = sample[1:]
#         features.append(X)
#         labels.append(y)
#
#     return features, labels

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.cv.CV_THRESH_BINARY_INV,cv_img)
    return cv_img

def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features,(-1,784))

    return features

if __name__ == "__main__":
    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    # 图片二值化
    features = binaryzation_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.8, random_state=23323)

    print 'end read'

    met = MaxEnt()
    met.train(rebuild_X(train_features), train_labels)

    print 'end train'

    test_predict = met.predict(rebuild_X(test_features))
    score = accuracy_score(test_labels,test_predict)

