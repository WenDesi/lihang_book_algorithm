import numpy as np
import struct
import matplotlib.pyplot as plt

from perceptron import *

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

if __name__=="__main__":
    imgs = loadImageSet()
    labels = loadLabelSet()

    index = 10

    print imgs[index]
    print labels[index]

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(imgs[index] , cmap='gray')
    # plt.show()

    perceptron = Perceptron()
    perceptron.hog_test(imgs[index])