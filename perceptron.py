#encoding=utf-8

import cv2
import numpy as np
import random

from extract_features import *

class Perceptron_Binary(object):
    study_step = 0.0001                                   # 学习步长
    study_total = 5000                                  # 学习次数
    feature_length = 34                                # hog特征维度
    trainset_size = 60000                               # 训练集大小
    testset_size = 10000

    object_num = 0                                      # 检测的目标
    model_filepath = 'model/perceptron_binary_model'    # 训练好的模型存储位置

    def init_parameters(self):
        self.w = np.zeros((self.feature_length,1))
        self.b = 0

    def train(self,train_imgs,train_labels):
        # 初始化w与b
        self.init_parameters()

        # 学习5000次
        study_count = 0
        kazhule = 100000
        csy = 0
        while True:
            csy += 1

            if csy > kazhule:
                print 'hehe'
                break

            # 随机选的数据
            index = random.randint(0,self.trainset_size-1)
            label = train_labels[index]

            # 计算yi(w*xi+b)
            yi = int(train_labels[index] == self.object_num) * 2 - 1
            result = yi * (np.dot(train_imgs[index],self.w) + self.b)

            img = np.reshape(train_imgs[index],(self.feature_length,1))

            # 如果yi(w*xi+b) <= 0 则更新 w 与 b 的值
            if result <= 0:
                self.w += img*yi*self.study_step
                self.b += yi*self.study_step

                study_count += 1
                print study_count
                if study_count > self.study_total:
                    break
                csy = 0

        model = np.append(self.w,np.array([self.b]))
        np.save(self.model_filepath,model)
        return self.w,self.b

    def load_model(self):
        model = np.load(self.model_filepath+'.npy')
        self.w = model[:-1]
        self.b = model[-1]

    def test(self,test_imgs,test_labels):
        # self.load_model()

        count = 0
        zero_count = 0
        for i in range(len(test_imgs)):
            img = test_imgs[i]
            label = test_labels[i]

            result = np.dot(img,self.w) + self.b
            result = result > 0

            if label == self.object_num:
                zero_count += 1
                if result:
                    count += 1

        print float(count)/float(zero_count)

    def load_model_test(self,w,b):
        self.w = w
        self.b = b

def train():
    features_filepath = 'features/train.vec.npy'
    features = np.load(features_filepath)

    labels = loadLabelSet()

    pb = Perceptron_Binary()
    w,b = pb.train(features,labels)
    return w,b

def test(w,b):
    features_filepath = 'features/test.vec.npy'
    features = np.load(features_filepath)

    labels = loadLabelSet()

    pb = Perceptron_Binary()
    pb.load_model_test(w,b)
    pb.test(features,labels)

if __name__ == '__main__':
    w,b = train()
    test(w,b)