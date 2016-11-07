# encoding=utf-8
# @Author: WenDesi
# @Date:   05-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 06-11-16


import math
import random

from collections import defaultdict



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

            for i in xrange(self.n):
                sigma = 1 / self.M * math.log(self.EPxy[i] / self.EPx[i])
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


def build_dataset(label,original_posins,radius,size):
    datasets = []
    dim = len(original_posins)

    for i in xrange(size):
        dataset = [label]
        for j in xrange(dim):
            point = random.randint(0,2*radius)-radius+original_posins[j]
            dataset.append(point)
        datasets.append(dataset)

    return datasets



def rebuild_features(features):
    new_features = []
    for feature in features:
        new_feature = []
        for i,f in enumerate(feature):
            new_feature.append(str(i)+'_'+str(f))
        new_features.append(new_feature)
    return new_features






if __name__ == "__main__":

    # 构建训练集
    trainset1 = build_dataset(0,[0,0],10,100)
    trainset2 = build_dataset(1,[30,30],10,100)

    trainset = trainset1
    trainset.extend(trainset2)
    random.shuffle(trainset)

    trainset_features = rebuild_features(map(lambda x:x[1:], trainset))
    trainset_labels = map(lambda x:x[0], trainset)

    # 训练
    met = MaxEnt()
    met.train(trainset_features,trainset_labels)

    # 构建测试集
    testset1 = build_dataset(0,[0,0],15,500)
    testset2 = build_dataset(1,[30,30],15,500)

    testset = testset1
    testset.extend(testset2)
    random.shuffle(testset)

    testset_features = rebuild_features(map(lambda x:x[1:], testset))
    testset_labels = map(lambda x:x[0], testset)

    # 测试
    testset_predicts = met.predict(testset_features)
    accuracy_score = float(len(filter(lambda x:x==True,[testset_labels[i]==testset_predicts[i] for i in xrange(len(testset_predicts))])))/float(len(testset_predicts))
    print "The accruacy socre is ", accuracy_score



