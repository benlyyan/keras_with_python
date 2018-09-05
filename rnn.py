"""
build rnn and lstm exercise 
"""

import numpy as np 
from keras import layers 
from keras import models 
from keras.preprocessing import sequence 
import os 


def sample_rnn_test():
    input_feat = 5
    out_feat = 3
    num_steps = 10
    seqs = np.random.random(size=(num_steps,input_feat))
    states = np.zeros((out_feat,))
    inter_states = []
    w = np.random.random((input_feat,out_feat))
    b = np.random.random((out_feat,))
    u = np.random.random((out_feat,out_feat))
    for seq in seqs:
        temp = np.tanh(np.dot(seq,w)+b+np.dot(states,u))
        inter_states.append(temp)
        states = temp 
    print(temp)
    print(np.concatenate(inter_states,axis=0).reshape((num_steps,out_feat)))
    # print(inter_states)
    # print(len(inter_states))


def build_lstm(num_feats,maxlen=500):
    model = models.Sequential()
    model.add(layers.Embedding(num_feats+1,64,input_length=maxlen))
    model.add(layers.LSTM(32,return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    return model 

def load_weather_data():
    filedir = 'D:\\dev\\ml\\competetion\\weather'
    filename = 'mpi_roof_2009a.csv'
    filepath = os.path.join(filedir,filename)
    f = open(filepath,encoding='ISO-8859-1')
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:-1]
    data = np.zeros((len(lines),21))
    for i,line in enumerate(lines):
        temp = [float(x) for x in line.split(',')[1:]]
        if len(temp)>0:
            data[i] = temp  
    return np.asarray(data)

def generator(data,min_ind,max_ind,lookback=720,delay=144,steps=6,batch_size=64,shuffle=False):
    num_samples,num_feats = data.shape 
    if (min_ind>=num_samples) or (max_ind+delay>num_samples) or (min_ind+lookback>max_ind):
        raise ValueError 
    # iterate generating batches from data 
    i = min_ind + lookback 
    while True:
        if shuffle:
            rows = np.random.randint(min_ind+lookback,max_ind,batch_size)
            np.random.shuffle(rows)
        else:
            if i + batch_size >= max_ind:
                i = min_ind + lookback 
            rows = np.arange(i,min(i+batch_size,max_ind))
            i += batch_size 
        batch = np.zeros((len(rows),lookback//steps,num_feats))
        label = np.zeros((len(rows),))
        for j,ind in enumerate(rows):
            batch[j] = data[(ind-lookback):ind:steps]
            label[j] = data[ind+delay][1]
        yield batch,label

def test_generator(min_ind,max_ind):
    x = np.random.random(size=(20000,10))
    # no shuffle
    gen = generator(x,min_ind,max_ind,shuffle=True)
    i = 0 
    for batch,label in gen:
        i += 1
        print(batch.shape,label.shape)
        if i >5:
            break 

def app_weather_prediction():
    lookback = 720 
    delay = 120 
    batch_size = 64 
    steps = 6 
    train_gen = generator(data,0,10000,lookback=lookback,batch_size=batch_size,delay=delay,steps=steps)
    val_gen = generator(data,10001,20000,lookback=lookback,batch_size=batch_size,delay=delay,steps=steps)
    val_steps = (20000-10001-lookback)//batch_size 
    model = models.Sequential() 
    model.add(layers.Flatten(input_shape(lookback//steps,data.shape[-1])))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(1,activation=None))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    model.fit_generator(train_gen,steps_per_epoch=500,epochs=10,validation_data=val_gen,validation_steps=val_steps)


if __name__=='__main__':
    # sample_rnn_test()
    # model = models.Sequential()
    # model.add(layers.Embedding(1000,10))
    # model.add(layers.SimpleRNN(32,return_sequences=True))
    # print(model.summary())

    # imdb using simple rnn 
    # path = 'D:\\dev\\ml\\competetion\\imdb\\'
    # imdb = np.load(path+'imdb.npz')
    # x = imdb['x_train']
    # y = imdb['y_train']
    # x_test = imdb['x_test']
    # y_test = imdb['y_test']
    # num_feats = max([max(sample) for sample in x])
    # x = sequence.pad_sequences(x,maxlen=500)
    # x_test = sequence.pad_sequences(x_test,maxlen=500)

    # model = models.Sequential()
    # model.add(layers.Embedding(num_feats+1,64,input_length=500))
    # model.add(layers.SimpleRNN(32,return_sequences=True))
    # model.add(layers.SimpleRNN(32))
    # model.add(layers.Dense(1,activation='sigmoid'))
    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    # model.fit(x,y,epochs=1,batch_size=64,validation_split=0.2)

    # imdb using lstm 
    # model = build_lstm(num_feats,maxlen=500)
    # model.fit(x,y,batch_size=64,epochs=10,validation_split=0.2)

    # test_generator(0,1000)
    data = load_weather_data()
    print(data.shape)


