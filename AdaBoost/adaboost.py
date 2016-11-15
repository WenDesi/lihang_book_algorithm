# encoding=utf-8
# @Author: wendesi
# @Date:   15-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   wendesi
# @Last modified time: 15-11-16

import math
import logging

class Sign(object):
    def __init__(self,features,labels,w):
        self.X = features
        self.Y = labels
        self.N = len(labels)

        self.w = w

        mmax = max(self.X)
        self.indexes = self.X[:]
        self.indexes.append(mmax+1)

    def _train_less_than_(self):
        index = -1
        error_score = 1000000

        for i in self.indexes:
            score = 0
            for j in xrange(self.N):
                val = -1
                if self.X[j]<i:
                    val = 1

                if val*self.Y[j]<0:
                    score += self.w[j]

            if score < error_score:
                index = i
                error_score = score

        return index,error_score



    def _train_more_than_(self):
        index = -1
        error_score = 1000000

        for i in self.indexes:
            score = 0
            for j in xrange(self.N):
                val = 1
                if self.X[j]<i:
                    val = -1

                if val*self.Y[j]<0:
                    score += self.w[j]

            if score < error_score:
                index = i
                error_score = score

        return index,error_score

    def train(self):
        less_index,less_score = self._train_less_than_()
        more_index,more_score = self._train_more_than_()

        if less_score < more_score:
            self.is_less = True
            self.index = less_index
            return less_score

        else:
            self.is_less = False
            self.index = more_index
            return more_score

    def predict(self,feature):
        if self.is_less:
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
        pass

    def _init_parameters_(self,features,labels):
        self.X = features
        self.Y = labels

        self.n = len(features[0])
        self.N = len(features)
        self.M = 100000                            # 分类器数目

        self.w = [1.0/self.N]*self.N
        self.alpha = []
        self.classifier = []

    def _w_(self,index,classifier,i):
        return self.w[i]*math.exp(-self.alpha[-1]*self.Y[i]*classifier.predict(self.X[i][index]))

    def _Z_(self,index,classifier):
        Z = 0

        for i in xrange(self.N):
            Z += self._w_(index,classifier,i)

        return Z

    def train(self,features,labels):

        self._init_parameters_(features,labels)

        for times in xrange(self.M):
            logging.debug('iterater %d' % times)

            best_classifier = (100000,None,None)        #(误差率,分类器,针对的特征)
            for i in xrange(self.n):
                features = map(lambda x:x[i],self.X)
                classifier = Sign(features,self.Y,self.w)
                error_score = classifier.train()

                if error_score < best_classifier[0]:
                    best_classifier = (error_score,i,classifier)

            em = best_classifier[0]
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

if __name__ == '__main__':
    print 'Start read data'

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]


    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print 'read data cost ',time_2 - time_1,' second','\n'

    print 'Start training'
    ada = AdaBoost()
    ada.train(train_features, train_labels)

    time_3 = time.time()
    print 'training cost ',time_3 - time_2,' second','\n'

    print 'Start predicting'
    test_predict = ada.predict(test_features)
    time_4 = time.time()
    print 'predicting cost ',time_4 - time_3,' second','\n'

    score = accuracy_score(test_labels,test_predict)
    print "The accruacy socre is ", score
