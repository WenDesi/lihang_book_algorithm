import pandas as pd
import numpy as np
import csv as csv
import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))
    print features.shape
    return features

train_raw=pd.read_csv('data/train_binary.csv',header=0)
train = train_raw.values


print 'Start PCA to 50'
train_x=train[0::,1::]
train_label=train[::,0]

features = get_hog_features(train_x)

# # pca
# pca = RandomizedPCA(n_components=50, whiten=True).fit(train_x)
# train_x_pca = pca.transform(train_x)
# test_x_pca = pca.transform(test)
# print train_x
a_train, b_train, a_label, b_label = train_test_split(features, train_label, test_size=0.33, random_state=23323)
print a_train.shape
print a_label.shape

print 'Start training'
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(a_train,a_label)
print 'Start predicting'
b_predict=rbf_svc.predict(b_train)
score=accuracy_score(b_label,b_predict)
print "The accruacy socre is ", score