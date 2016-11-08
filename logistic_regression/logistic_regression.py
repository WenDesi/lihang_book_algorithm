# encoding=utf-8
# @Author: WenDesi
# @Date:   08-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 08-11-16

import math
import random


def predict_(x, w):
    wx = sum([w[j] * x[j] for j in xrange(len(w))])
    exp_wx = math.exp(wx)

    predict1 = exp_wx / (1 + exp_wx)
    predict0 = 1 / (1 + exp_wx)

    if predict1 > predict0:
        return 1
    else:
        return 0


def train(features, labels):
    w = [0.0] * (len(features[0]) + 1)

    learning_step = 0.00001
    max_iteration = 1000
    correct_count = 0
    time = 0

    while time < max_iteration:
        index = random.randint(0, len(labels) - 1)
        x = features[index]
        x.append(1.0)
        y = labels[index]

        if y == predict_(x, w):
            correct_count += 1
            if correct_count > max_iteration:
                break
            continue

        print 'iterater times %d' % time
        time += 1
        correct_count = 0

        wx = sum([w[i] * x[i] for i in xrange(len(w))])
        exp_wx = math.exp(wx)

        for i in xrange(len(w)):
            w[i] -= learning_step * (-y * x[i] + float(x[i] * exp_wx) / float(1 + exp_wx))

    return w


def predict(features, w):
    labels = []

    for feature in features:
        feature.append(1)
        x = feature

        labels.append(predict_(x,w))

    return labels


def build_dataset(label, original_posins, radius, size):
    datasets = []
    dim = len(original_posins)

    for i in xrange(size):
        dataset = [label]
        for j in xrange(dim):
            point = random.randint(0, 2 * radius) - radius + original_posins[j]
            dataset.append(point)
        datasets.append(dataset)

    return datasets

if __name__ == "__main__":

    # 构建训练集
    trainset1 = build_dataset(0, [0, 0], 10, 100)
    trainset2 = build_dataset(1, [30, 30], 10, 100)

    trainset = trainset1
    trainset.extend(trainset2)
    random.shuffle(trainset)

    trainset_features = map(lambda x: x[1:], trainset)
    trainset_labels = map(lambda x: x[0], trainset)

    # 训练
    w = train(trainset_features, trainset_labels)

    # 构建测试集
    testset1 = build_dataset(0, [0, 0], 10, 500)
    testset2 = build_dataset(1, [30, 30], 10, 500)

    testset = testset1
    testset.extend(testset2)
    random.shuffle(testset)

    testset_features = map(lambda x: x[1:], testset)
    testset_labels = map(lambda x: x[0], testset)

    # 测试
    testset_predicts = predict(testset_features, w)
    print 'asad'
    accuracy_score = float(len(filter(lambda x: x == True, [testset_labels[i] == testset_predicts[
                           i] for i in xrange(len(testset_predicts))]))) / float(len(testset_predicts))
    print "The accruacy socre is ", accuracy_score
