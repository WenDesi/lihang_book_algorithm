#encoding=utf-8

import cv2
import time
import math
import numpy as np
import pandas as pd


from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

total_class = 10

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


class Tree(object):
    def __init__(self,node_type,Class = None, feature = None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add_tree(self,val,tree):
        self.dict[val] = tree

    def predict(self,features):
        if self.node_type == 'leaf':
            return self.Class

        print 'in'

        tree = self.dict[features[self.feature]]
        return tree.predict(features)

def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """

    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent

def calc_ent_grap(x,y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap

def train(train_set,train_label,features,epsilon):
    global total_class

    LEAF = 'leaf'
    INTERNAL = 'internal'


    # 步骤1——如果train_set中的所有实例都属于同一类Ck
    label_dict = [0 for i in xrange(total_class)]
    for label in train_label:
        label_dict[label] += 1

    for label, label_count in enumerate(label_dict):
        if label_count == len(train_label):
            tree = Tree(LEAF,Class = label)
            return tree

    # 步骤2——如果features为空
    max_len,max_class = 0,0
    for i in xrange(total_class):
        class_i = filter(lambda x:x==i,train_label)
        if len(class_i) > max_len:
            max_class = i
            max_len = len(class_i)

    if len(features) == 0:
        tree = Tree(LEAF,Class = max_class)
        return tree

    # 步骤3——计算信息增益
    max_feature = 0
    max_gda = 0

    D = train_label
    HD = calc_ent(D)
    for feature in features:
        A = np.array(train_set[:,feature].flat)
        gda = HD - calc_condition_ent(A,D)

        if gda > max_gda:
            max_gda,max_feature = gda,feature

    # 步骤4——小于阈值
    if max_gda < epsilon:
        tree = Tree(LEAF,Class = max_class)
        return tree

    # 步骤5——构建非空子集
    sub_features = filter(lambda x:x!=max_feature,features)
    tree = Tree(INTERNAL,feature=max_feature)

    feature_col = np.array(train_set[:,max_feature].flat)
    feature_value_list = set([feature_col[i] for i in range(feature_col.shape[0])])
    for feature_value in feature_value_list:

        index = []
        for i in xrange(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = train(sub_train_set,sub_train_label,sub_features,epsilon)
        tree.add_tree(feature_value,sub_tree)

    return tree

def predict(test_set,tree):

    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)



if __name__ == '__main__':
    # classes = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
    #
    # age = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    # occupation = [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0]
    # house = [0,0,0,1,0,0,0,1,1,1,1,1,0,0,0]
    # loan = [0,1,1,0,0,0,1,1,2,2,2,1,1,2,0]
    #
    # features = []
    #
    # for i in range(15):
    #     feature = [age[i],occupation[i],house[i],loan[i]]
    #     features.append(feature)
    #
    # trainset = np.array(features)
    #
    # tree = train(trainset,np.array(classes),[0,1,2,3],0.1)
    #
    # print type(tree)
    # features = [0,0,0,1]
    # print tree.predict(np.array(features))


    print 'Start read data'

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    features = binaryzation_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print 'read data cost ',time_2 - time_1,' second','\n'

    print 'Start training'
    tree = train(train_features,train_labels,[i for i in range(784)],0.2)
    print type(tree)
    print 'knn do not need to train'
    time_3 = time.time()
    print 'training cost ',time_3 - time_2,' second','\n'

    print 'Start predicting'
    test_predict = predict(test_features,tree)
    time_4 = time.time()
    print 'predicting cost ',time_4 - time_3,' second','\n'

    score = accuracy_score(test_labels,test_predict)
    print "The accruacy socre is ", score











