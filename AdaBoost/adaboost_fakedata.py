# encoding=utf-8
# @Author: wendesi
# @Date:   15-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   wendesi
# @Last modified time: 15-11-16

import logging

from generate_dataset import *
from adaboost import AdaBoost

from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_features, train_labels, test_features, test_labels = generate_dataset(200)

    ada = AdaBoost()
    ada.train(train_features,train_labels)

    print 'end train'
    test_predict = ada.predict(test_features)


    score = accuracy_score(test_labels,test_predict)
    print "ada boost the accruacy socre is ", score
