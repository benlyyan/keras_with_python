# -*- coding:utf-8 -*-

from keras import layers,models
from keras.utils import to_categorical  
import csv
import numpy as np  
import os 

def read_csv(path):
    with open(path,'r') as f:
        data = csv.reader(f)
        res = []
        for line in data:
            res.append(line)
    data = np.array(res[1:])
    label = data[:,0].astype('float32')
    data = data[:,1:].astype('float32')/255. 
    return data,label 

def load_dataset(path_train,path_test):
    x,y = read_csv(path_train)
    testx,testy = read_csv(path_test)
    return x,y,testx,testy 

def Digitmodel():
    model = models.Sequential()
    model.add(layers.Dense(512,activation='relu',input_shape=(784,)))
    model.add(layers.Dense(10,activation='softmax'))
    return model 

if __name__ =="__main__":
    path = "D:\dev\ml\competetion\digit"
    train_path = os.path.join(path,'train.csv')
    test_path = os.path.join(path,'test.csv')
    x,y,testx,testy = load_dataset(train_path,test_path)
    x = x.astype('float32')/255. 
    x = x[:2000]
    testx = testx.astype('float32')/255. 
    y = to_categorical(y)
    y = y[:2000]
    testy = to_categorical(testy)
    model = Digitmodel()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    model.fit(x,y,epochs=5,batch_size=64)

