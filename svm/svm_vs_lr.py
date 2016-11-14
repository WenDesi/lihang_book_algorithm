# encoding=utf-8
# @Author: WenDesi
# @Date:   12-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 13-11-16

import time
import csv
import random
import logging

from sklearn.metrics import accuracy_score
from generate_dataset import *
from svm import SVM
from logistic_regression import LogisticRegression
from sklearn import svm


if __name__ == "__main__":


    writer = csv.writer(file('svm_vs_lr.csv', 'ab'))

    # for i in xrange(10):
    #     print 'competition now in lap %d' % i

    my_svm1 = SVM()
    my_svm2 = SVM(kernel='poly')

    lr = LogisticRegression()

    train_features, train_labels, test_features, test_labels = generate_dataset(2000,visualization=False)

    my_svm1.train(train_features,train_labels)
    my_svm2.train(train_features,train_labels)

    train_labels = map(lambda x:(x+1)/2,train_labels)
    lr.train(train_features,train_labels)

    result = []

    predict = my_svm1.predict(test_features)
    score=accuracy_score(test_labels,predict)
    result.append(score)

    predict = my_svm2.predict(test_features)
    score=accuracy_score(test_labels,predict)
    result.append(score)

    predict = lr.predict(test_features)
    test_labels = map(lambda x:(x+1)/2,test_labels)
    score=accuracy_score(test_labels,predict)
    result.append(score)

    writer.writerow(result)
