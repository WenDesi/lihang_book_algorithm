import pandas as pd
import numpy as np
import csv as csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
train_raw=pd.read_csv('data/train.csv',header=0)
test_raw=pd.read_csv('data/test.csv',header=0)
train = train_raw.values
test = test_raw.values
print 'Start PCA to 50'
train_x=train[0::,1::]
train_label=train[::,0]
pca = RandomizedPCA(n_components=50, whiten=True).fit(train_x)
train_x_pca=pca.transform(train_x)
test_x_pca=pca.transform(test)
a_train, b_train, a_label, b_label = train_test_split(train_x_pca, train_label, test_size=0.33, random_state=23323)
print a_train.shape
print a_label.shape
print 'Start training'
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(a_train,a_label)
print 'Start predicting'
b_predict=rbf_svc.predict(b_train)
score=accuracy_score(b_label,b_predict)
print "The accruacy socre is ", score
print 'Start writing!'
out=rbf_svc.predict(test_x_pca)
n,m=test_x_pca.shape
ids = range(1,n+1)
predictions_file = open("out3.csv","wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids,out))
predictions_file.close()
print 'All is done'