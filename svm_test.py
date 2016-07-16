#encoding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 训练的点数
    train_pts = 30

    # 创建测试的数据点，2类
    # 以(-1.5, -1.5)为中心
    rand1 = np.ones((train_pts,2)) * (-2) + np.random.rand(train_pts, 2)
    print('rand1：')
    print(rand1)

    # 以(1.5, 1.5)为中心
    rand2 = np.ones((train_pts,2)) + np.random.rand(train_pts, 2)
    print('rand2:')
    print(rand2)

    # 合并随机点，得到训练数据
    train_data = np.vstack((rand1, rand2))
    train_data = np.array(train_data, dtype='float32')
    train_label = np.vstack( (np.zeros((train_pts,1), dtype='int32'), np.ones((train_pts,1), dtype='int32')))

    # 显示训练数据
    plt.figure(1)
    plt.plot(rand1[:,0], rand1[:,1], 'o')
    plt.plot(rand2[:,0], rand2[:,1], 'o')
    plt.plot(rand2[:,0], rand2[:,1], 'o')

    # 创建分类器
    svm = cv2.SVM()
    params = dict( kernel_type = cv2.SVM_LINEAR,svm_type = cv2.SVM_C_SVC,C = 1 )
    svm.train(train_data,train_label,params = params)

    print train_data.shape
    print train_label.shape

    # 测试数据，20个点[-2,2]
    pt = np.array(np.random.rand(20,2) * 4 - 2, dtype='float32')
    print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
    print pt
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    res = np.float32( [svm.predict(s) for s in pt])
    print "res = "
    print res

    # 按label进行分类显示
    plt.figure(2)


    llist1,llist2 = [],[]
    for i in range(len(pt)):
        if res[i] > 0.5:
            llist2.append(pt[i])
        else:
            llist1.append(pt[i])

    # 第一类
    type_data = np.array(llist1)
    plt.plot(type_data[:,0], type_data[:,1], 'o')

    # 第二类
    type_data = np.array(llist2)
    plt.plot(type_data[:,0], type_data[:,1], 'o')

    plt.show()