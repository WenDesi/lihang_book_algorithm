# encoding=utf-8
# @Author: wendesi
# @Date:   15-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   wendesi
# @Last modified time: 15-11-16

import cv2
import time
import math
import ctypes
import logging
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

sign_time_count = 0

class Sign(object):
    def __init__(self,is_less,index):
        self.is_less = is_less
        self.index = index


    def predict(self,feature):
        if self.is_less>0:
            if feature<self.index:
                return 1.0
            else:
                return -1.0
        else:
            if feature<self.index:
                return -1.0
            else:
                return 1.0


class AdaBoost(object):

    def __init__(self):
        ll = ctypes.cdll.LoadLibrary
        self.lib = ll("Sign/x64/Release/Sign.dll")

    def rebuild_X(self,X):
        length = self.n*self.N
        self.X_matrix = (ctypes.c_int * length)()

        for i in xrange(self.n):
            for j in xrange(self.N):
                self.X_matrix[i*self.N+j] = X[j][i]

    def rebuild_Y(self,Y):
        self.C_Y = (ctypes.c_int * self.N)()
        for i in xrange(self.N):
            self.C_Y[i] = Y[i]


    def _init_parameters_(self,features,labels):
        self.Y = labels

        self.n = len(features[0])
        self.N = len(features)
        self.M = 100                            # 分类器数目

        self.w = [1.0/self.N]*self.N
        self.alpha = []
        self.classifier = []

        self.rebuild_X(features)
        self.rebuild_Y(labels)

    def _w_(self,index,classifier,i):
        feature = self.X_matrix[index*self.N+i]
        return self.w[i]*math.exp(-self.alpha[-1]*self.Y[i]*classifier.predict(feature))

    def _Z_(self,index,classifier):
        Z = 0

        for i in xrange(self.N):
            Z += self._w_(index,classifier,i)

        return Z

    def build_c_w(self):
        C_w = (ctypes.c_double * self.N)()
        for i in xrange(self.N):
            C_w[i] = ctypes.c_double(self.w[i])
        return C_w

    def train(self,features,labels):

        self._init_parameters_(features,labels)

        for times in xrange(self.M):
            logging.debug('iterater %d' % times)


            C_w = self.build_c_w()
            min_error = ctypes.c_double(100000)
            is_less = ctypes.c_int(-1)
            feature_index = ctypes.c_int(-1)

            index = self.lib.find_min_error(self.X_matrix,self.n,self.N,self.C_Y,C_w,ctypes.byref(min_error),ctypes.byref(is_less),ctypes.byref(feature_index))



            em = min_error.value
            best_classifier = (em,feature_index.value,Sign(is_less.value,index))        #(误差率,针对的特征,分类器)
            print 'em is %s, index is %s' % (str(em),str(feature_index.value))


            if em==0:
                self.alpha.append(100)
            else:
                self.alpha.append(0.5*math.log((1-em)/em))

            self.classifier.append(best_classifier[1:])

            Z = self._Z_(best_classifier[1],best_classifier[2])

            for i in xrange(self.N):
                self.w[i] = self._w_(best_classifier[1],best_classifier[2],i)/Z

    def _predict_(self,feature):

        result = 0.0
        for i in xrange(self.M):
            index = self.classifier[i][0]
            classifier = self.classifier[i][1]

            result += self.alpha[i]*classifier.predict(feature[index])

        if result>0:
            return 1
        return -1



    def predict(self,features):
        results = []

        for feature in features:
            results.append(self._predict_(feature))

        return results

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

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    print 'Start read data'

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]


    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    features = binaryzation_features(imgs)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print 'read data cost ',time_2 - time_1,' second','\n'

    print 'Start training'
    train_labels = map(lambda x:2*x-1,train_labels)
    ada = AdaBoost()
    ada.train(train_features, train_labels)

    time_3 = time.time()
    print 'training cost ',time_3 - time_2,' second','\n'

    print 'Start predicting'
    test_predict = ada.predict(test_features)
    time_4 = time.time()
    print 'predicting cost ',time_4 - time_3,' second','\n'

    test_labels = map(lambda x:2*x-1,test_labels)
    score = accuracy_score(test_labels,test_predict)
    print "The accruacy socre is ", score
