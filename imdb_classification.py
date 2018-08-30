# -*- coding:utf-8 -*-

import numpy as np  
from keras import layers 
from keras import models 

def seq_to_matrix(sequences,maxlen=10000):
    res = np.zeros((len(sequences),maxlen))
    for i,seq in enumerate(sequences):
        res[i,seq] = 1 
    return res  


def count_res(tol,rmblist=[0,2,4,2,1,0]):
    count = 0
    rmb = np.array([1,5,10,20,50,100])
    amount = rmb.dot(np.array(rmblist))
    if amount < tol:
        return 0 
    for i0 in range(rmblist[0]+1):
        for i1 in range(rmblist[1]+1):
            for i2 in range(rmblist[2]+1):
                for i3 in range(rmblist[3]+1):
                    for i4 in range(rmblist[4]+1):
                        for i5 in range(rmblist[5]+1):
                            tmplist = np.array([i0,i1,i2,i3,i4,i5])
                            tmp = tmplist.dot(rmb)
                            if tmp == tol:
                                count += 1
    return count 

if __name__=="__main__":
    # count = count_res(100,rmblist=[0,0,5,6,3,2])
    # print(count)
    # model = models.Sequential() 
    # model.add(layers.Conv2D(32,(3,3),input_shape=(28,28,3),padding='same'))
    # model.add(layers.BatchNormalization(axis=3))
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D((2,2),strides=(1,1)))
    # model.add(layers.Conv2D(64,(3,3),activation='relu'))
    # model.add(layers.MaxPooling2D((2,2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64,activation='relu'))
    # model.add(layers.Dense(10,activation='softmax'))
    # print(model.summary())
    x_input = layers.Input((28,28,3))
    X = layers.Conv2D(32,(3,3),activation='relu')(x_input)
    X = layers.MaxPooling2D((2,2))(X)
    X = layers.Conv2D(64,(3,3),activation='relu')(X)
    X = layers.MaxPooling2D((2,2))(X)
    X = layers.Flatten()(X)
    X = layers.Dense(32,activation='relu')(X)
    X = layers.Dense(10,activation='softmax')(X)
    model = models.Model(inputs=x_input,outputs=X)
    print(model.summary())


