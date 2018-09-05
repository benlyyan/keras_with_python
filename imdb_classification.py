# -*- coding:utf-8 -*-

import numpy as np  
from keras import layers 
from keras import models 
from keras.preprocessing import sequence 
from keras import models 
import os 
from keras import optimizers 

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

def test():
    texts = ['this is an apple which is big','that is an orange which is samll']
    max_len = 10 
    dict_val = {}
    for sample in texts:
        for i,word in enumerate(sample.split()):
            if word not in dict_val:
                dict_val[word] = len(dict_val)+1 
    res = np.zeros((len(texts),max_len,max(dict_val.values())+1))
    for i,sample in enumerate(texts):
        for j,word in list(enumerate(sample.split()))[:max_len]:
            index = dict_val.get(word)
            res[i,j,index] = 1
    return res 

from keras.preprocessing.text import Tokenizer
def test1():
    texts = ['this is an apple which is big','that is an orange which is samll']
    token = Tokenizer(num_words=10)
    token.fit_on_texts(texts)
    seq = token.texts_to_sequences(texts)
    mat = token.texts_to_matrix(texts,mode='binary')
    print(seq)
    print(mat)

def load_imdb_rawdata():
    path = 'D:\\dev\\ml\\competetion\\imdb\\aclImdb'
    train_dir = os.path.join(path,'train')
    res = []
    y = []
    for label_tag in ['neg','pos']:
        train_path = os.path.join(train_dir,label_tag)
        for fname in os.listdir(train_path):
            if fname[-4:] =='.txt':
                with open(os.path.join(train_path,fname),encoding='utf-8') as f:
                    res.append(f.read())
                if label_tag == 'neg':
                    y.append(0)
                else:
                    y.append(1)
    return res,y

def token(maxlen=20):
    x,y = load_imdb_rawdata()
    token = Tokenizer(num_words=10000)
    token.fit_on_texts(x)
    seqs = token.texts_to_sequences(x)
    seqs = sequence.pad_sequences(seqs,maxlen=maxlen)
    index = np.arange(len(y))
    np.random.shuffle(index)
    seqs = seqs[index]
    y = np.asarray(y)[index]
    word_index = token.word_index 
    return seqs,y,word_index 

def load_pretrained_weights():
    path = 'D:\\dev\\ml\\competetion\\imdb\\aclImdb'
    file_path = os.path.join(path,'glove.6B.50d.txt')
    word_ind = {}
    with open(file_path,encoding='utf-8') as f:
        for line in f:
            line = line.split()
            word = line[0]
            weigths = line[1:]
            weigths = np.asarray(weigths,dtype='float32')
            word_ind[word] = weigths
    return word_ind

def build_embedding_model(num_words=10000,embedding_dim=50,maxlen=50):
    x,y,word_ind = token(maxlen=maxlen)
    x_train,y_train = x[:200],y[:200]
    x_val,y_val = x[200:12000],y[200:12000]
    weigths = load_pretrained_weights()
    weigths_matrix = np.zeros((num_words,embedding_dim))
    for word,ind in word_ind.items():
        if ind < num_words:
            if word in weigths:
                weigths_matrix[ind] = weigths[word]
    model = models.Sequential()
    model.add(layers.Embedding(num_words,embedding_dim,input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.layers[0].set_weights([weigths_matrix])
    model.layers[0].trainable = False 
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),loss='binary_crossentropy',metrics=['acc'])
    model.fit(x,y,epochs=20,batch_size=64,validation_data=(x_val,y_val))
    model.save_weights('imdb_with_pretrained_weights.h5')
    return model 

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
    # x_input = layers.Input((28,28,3))
    # X = layers.Conv2D(32,(3,3),activation='relu')(x_input)
    # X = layers.MaxPooling2D((2,2))(X)
    # X = layers.Conv2D(64,(3,3),activation='relu')(X)
    # X = layers.MaxPooling2D((2,2))(X)
    # X = layers.Flatten()(X)
    # X = layers.Dense(32,activation='relu')(X)
    # X = layers.Dense(10,activation='softmax')(X)
    # model = models.Model(inputs=x_input,outputs=X)
    # print(model.summary())
    # print(test1())

    # embedding 
    # path = 'D:\\dev\\ml\\competetion\\imdb\\'
    # imdb = np.load(path+'imdb.npz')
    # x = imdb['x_train']
    # y = imdb['y_train']
    # x_test = imdb['x_test']
    # y_test = imdb['y_test']
    # num_feats = max([max(sample) for sample in x])
    # x = sequence.pad_sequences(x,maxlen=20)
    # x_test = sequence.pad_sequences(x_test,maxlen=20)
    # model = models.Sequential()
    # model.add(layers.Embedding(num_feats,8,input_length=20))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1,activation='sigmoid'))
    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    # model.summary()
    # hist = model.fit(x,y,epochs=10,batch_size=32,validation_split=0.2)

    # pre-trained embedding 
    build_embedding_model()
    # weigths = load_pretrained_weights()
    # print(weigths['this'])





