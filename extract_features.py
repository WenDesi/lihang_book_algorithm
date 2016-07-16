#encoding=utf-8

import numpy as np
import cv2
import time
import struct
import matplotlib.pyplot as plt


def loadImageSet(which=0):
    print "load image set"
    binfile=None
    if which==0:
        binfile = open("data/train-images.idx3-ubyte", 'rb')
    else:
        binfile=  open("data/t10k-images.idx3-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII' , buffers ,0)
    print "head,",head

    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]
    #[60000]*28*28
    bits=imgNum*width*height
    bitsString='>'+str(bits)+'B' #like '>47040000B'

    imgs=struct.unpack_from(bitsString,buffers,offset)

    binfile.close()
    imgs=np.reshape(imgs,[imgNum,width,height])
    print "load imgs finished"
    return imgs

def loadLabelSet(which=0):
    print "load label set"
    binfile=None
    if which==0:
        binfile = open("data/train-labels.idx1-ubyte", 'rb')
    else:
        binfile=  open("data/t10k-labels.idx1-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II' , buffers ,0)
    print "head,",head
    imgNum=head[1]

    offset = struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels= struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels=np.reshape(labels,[imgNum,1])

    #print labels
    print 'load label finished'
    return labels

def get_features(imgs):
    features = []
    hog = cv2.HOGDescriptor('hog.xml')

    # 二值化
    for i in range(len(imgs)):
        cv_img = imgs[i].astype(np.uint8)
        cv2.threshold(cv_img,25,255,cv2.cv.CV_THRESH_BINARY_INV,imgs[i])

    for img in imgs:
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        hog_feature = np.transpose(hog_feature)

        features.append(hog_feature)

    return np.array(features)

def get_hog_features():
    # trainset features
    features_filepath = 'features/train.vec.npy'

    imgs = loadImageSet()
    labels = loadLabelSet()

    features = get_features(imgs)
    np.save(features_filepath,features)

    # testset features
    features_filepath = 'features/test.vec.npy'

    imgs = loadImageSet(1)
    labels = loadLabelSet(1)

    features = get_features(imgs)
    np.save(features_filepath,features)

    features = np.load(features_filepath)

def manul_features(imgs):
    features = []

    tt = 0
    for img in imgs:
        print tt
        tt += 1
        feature = []
        cv_img = img.astype(np.uint8)
        cv2.threshold(cv_img,25,255,cv2.cv.CV_THRESH_BINARY_INV,cv_img)

        range_list = [[0,7,0,7],
                      [0,7,7,14],
                      [0,7,14,21],
                      [0,7,21,28],
                      [7,14,0,7],
                      [7,11,7,11],
                      [7,11,11,14],
                      [7,11,14,17],
                      [7,11,17,21],
                      [11,14,7,11],
                      [11,14,11,14],
                      [11,14,14,17],
                      [11,14,17,21],
                      [7,14,21,28],
                      [14,21,21,28],
                      [21,28,21,28],
                      [14,21,0,7],
                      [21,28,0,7],
                      [14,17,7,11],
                      [14,17,11,14],
                      [14,17,14,17],
                      [14,17,17,21],
                      [17,21,7,11],
                      [17,21,11,14],
                      [17,21,14,17],
                      [17,21,17,21],
                      [21,24,7,11],
                      [21,24,11,14],
                      [21,24,14,17],
                      [21,24,17,21],
                      [24,28,7,11],
                      [24,28,11,14],
                      [24,28,14,17],
                      [24,28,17,21]]



        for range_ in range_list:
            count = 0
            for i in range(range_[0],range_[1]):
                for j in range(range_[2],range_[3]):
                    if cv_img[i][j] < 50:
                        count += 1
            feature.append(count)
        features.append(feature)

    return np.array(features)

def get_manual_features():
    trainset_features_filepath = 'features/train.vec.npy'
    testset_features_filepath = 'features/test.vec.npy'

    imgs = loadImageSet()
    features = manul_features(imgs)
    np.save(trainset_features_filepath,features)

    imgs = loadImageSet(1)
    features = manul_features(imgs)
    np.save(testset_features_filepath,features)

if __name__=="__main__":
    get_manual_features()
    # get_hog_features()




