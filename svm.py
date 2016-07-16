#encoding=utf-8

import cv2
import numpy as np
import random

from extract_features import *

class SVM(object):
    study_step = 0.0001                                   # 学习步长
    study_total = 50000                                  # 学习次数
    feature_length = 34                                # hog特征维度
    trainset_size = 60000                               # 训练集大小
    testset_size = 10000

    object_num = 0                                      # 检测的目标
    model_filepath = 'model/perceptron_binary_model'    # 训练好的模型存储位置

    def init_parameters(self):
        self.w = np.zeros((self.feature_length,1))
        self.b = 0

    def train(self,train_imgs,train_labels):

        labels = []
        for label in train_labels:
            labels.append(float(label == self.object_num))
        labels = np.array(labels)


        # 创建分类器
        trainData = np.float32(train_imgs)
        responses = np.float32(labels)
        svm = cv2.SVM()
        params = dict( kernel_type = cv2.SVM_LINEAR,svm_type = cv2.SVM_C_SVC,C = 1 )
        svm.train(trainData,responses,params = params)

        self.svm = svm

    def load_model(self):
        model = np.load(self.model_filepath+'.npy')
        self.w = model[:-1]
        self.b = model[-1]

    def test(self,test_imgs,test_labels):
        # self.load_model()

        testData = np.float32(test_imgs)

        count = 0
        zero_count = 0
        for i in range(len(testData)):
            img = testData[i]
            label = test_labels[i]


            result = self.svm.predict(img)

            if label == self.object_num:
                zero_count += 1
                if result == 1:
                    count += 1

            print 'label:',label,'  result:',result
        print float(count)/zero_count

    def load_model_test(self,w,b):
        self.w = w
        self.b = b

def train():
    features_filepath = 'features/train.vec.npy'
    features = np.load(features_filepath)

    labels = loadLabelSet()

    pb = SVM()
    pb.train(features,labels)
    return pb

def test(pb):
    features_filepath = 'features/test.vec.npy'
    features = np.load(features_filepath)

    labels = loadLabelSet()

    pb.test(features,labels)

if __name__ == '__main__':
    pb = train()
    test(pb)