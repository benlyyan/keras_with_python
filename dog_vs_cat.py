# -*- coding:utf-8 -*- 

import os
import shutil 
from keras import layers 
from keras import models 
from keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt 
import numpy as np 
from keras import optimizers   
from keras import initializers 

def build_model(input_shape=(64,64,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32,(7,7),strides=(2,2),padding='same',input_shape=input_shape))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3)))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3)))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(256,(3,3)))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model 

def model_without_augmentation(train_dir,val_dir,model):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x150
    target_size=(150, 150),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
    history = model.fit_generator(train_generator,steps_per_epoch=100,
        epochs=20,validation_data=validation_generator,
        validation_steps=50)
    model.save('cats_dogs_samll.h5')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc)+1)
    plt.plot(epochs,acc,'bo',label='training acc')
    plt.plot(epochs,val_acc,'b',label='validation acc')
    plt.legend()
    plt.figure()
    plt.plot(epochs,loss,'bo',label='train loss')
    plt.plot(epochs,val_loss,'b',label='val loss')
    plt.legend()
    plt.show()

def model_with_augmentation(train_dir,val_dir,model):
    train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,
        height_shift_range=0.2,rotation_range=40,
        zoom_range=0.2,horizontal_flip=True,shear_range=0.2,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,batch_size=20,target_size=(150,150),class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(val_dir,batch_size=20,target_size=(150,150),class_mode='binary')
    history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=20,validation_data=validation_generator,validation_steps=50)
    model.save('vgg16_cats_dogs_samll_dataaugmentation.h5')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc)+1)
    plt.plot(epochs,acc,'bo',label='training acc')
    plt.plot(epochs,val_acc,'b',label='validation acc')
    plt.legend()
    plt.figure()
    plt.plot(epochs,loss,'bo',label='train loss')
    plt.plot(epochs,val_loss,'b',label='val loss')
    plt.legend()
    plt.show()  

def vgg16_extract_feature(vgg16,data_dir,num_samples,output_shape,batch_size=20):
    datadagen = ImageDataGenerator(rescale=1./255)
    train_generator = datadagen.flow_from_directory(data_dir,batch_size=batch_size,target_size=(150,150),class_mode='binary')
    extracted_feature = np.zeros((num_samples,)+output_shape)
    labels = np.zeros(num_samples)
    i = 0 
    for x_batch,y_batch in train_generator:
        temp_feature = vgg16.predict(x_batch)
        extracted_feature[i*batch_size:(i+1)*batch_size] = temp_feature 
        labels[i*batch_size:(i+1)*batch_size] = y_batch 
        i += 1 
        if i*batch_size >= num_samples:
            break 
    return extracted_feature,labels 

if __name__=="__main__":
    # create train test val dataset 
    train = 'dogs_cats'
    val_dir = os.path.join(train,'validation')
    # os.mkdir(val_dir)
    val_dog_dir = os.path.join(val_dir,'dog')
    # os.mkdir(val_dog_dir)
    val_cat_dir = os.path.join(val_dir,'cat')
    # os.mkdir(val_cat_dir)

    train_dog = 'dog'
    train_dir = os.path.join(train,'train')
    train_dog_dir = os.path.join(train_dir,'dog')
    # os.mkdir(train_dog_dir)
    train_cat_dir = os.path.join(train_dir,'cat')
    # os.mkdir(train_cat_dir)
    test_dir = os.path.join(train,'test1')
    test_dog_dir = os.path.join(test_dir,'dog')
    # os.mkdir(test_dog_dir)
    test_cat_dir = os.path.join(test_dir,'cat')
    # os.mkdir(test_cat_dir)
    test_dog_files = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
    test_cat_files = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
    # for i,j in zip(test_dog_files,test_cat_files):
    #     src = os.path.join(train_dir,i)
    #     dst = os.path.join(test_dog_dir,i)
    #     shutil.move(src,dst)
    #     src = os.path.join(train_dir,j)
    #     dst = os.path.join(test_cat_dir,j)
    #     shutil.move(src,dst)

    # train_dog_files = ['dog.{}.jpg'.format(i) for i in range(1000)]
    # for file in train_dog_files:
    #     src = os.path.join(train_dir,file)
    #     dst = os.path.join(train_dog_dir,file)
    #     shutil.move(src,dst)
    # 1000 pictures for training 
    # train_cat_files = ['cat.{}.jpg'.format(i) for i in range(1000)]
    # for file in train_cat_files:
    #     src = os.path.join(train_dir,file)
    #     dst = os.path.join(train_cat_dir,file)
    #     shutil.move(src,dst)
    # val_dog_files = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
    # val_cat_files = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
    # for i in range(len(val_dog_files)):
    #     src = os.path.join(train_dir,val_dog_files[i])
    #     dst = os.path.join(val_dog_dir,val_dog_files[i])
    #     shutil.move(src,dst)
    #     src = os.path.join(train_dir,val_cat_files[i])
    #     dst = os.path.join(val_cat_dir,val_cat_files[i])
    #     shutil.move(src,dst)

    # model = build_model(input_shape=(150,150,3))
    # print(model.summary())
    # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    # model_with_augmentation(train_dir,val_dir,model)

    # extract feature : without data augmentation
    convbase = models.load_model('vgg16_without_top.h5')
    # trainx,trainy = vgg16_extract_feature(convbase,train_dir,num_samples=1000,output_shape=(4,4,512),batch_size=20)
    # valx,valy = vgg16_extract_feature(convbase,val_dir,num_samples=500,output_shape=(4,4,512),batch_size=20)
    # testx,testy = vgg16_extract_feature(convbase,test_dir,num_samples=500,output_shape=(4,4,512),batch_size=20)
    # # print(trainx.shape,trainy.shape)
    # trainx = np.reshape(trainx,(trainx.shape[0],-1))
    # valx = np.reshape(valx,(valx.shape[0],-1))
    # testx = np.reshape(testx,(testx.shape[0],-1))

    # model = models.Sequential()
    # model.add(layers.Dense(256,activation='relu',input_shape=(trainx.shape[1],)))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(1,activation='sigmoid'))
    # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    # history = model.fit(trainx,trainy,batch_size=20,epochs=20,validation_data=(valx,valy))
    # res = model.evaluate(testx,testy)
    # print(res) 

    # feature extraction : with data augmentation 
    # model_fe = models.Sequential()
    # model_fe.add(convbase)
    # model_fe.add(layers.Flatten())
    # model_fe.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=0)))
    # model_fe.add(layers.Dense(1,activation='sigmoid',kernel_initializer=initializers.glorot_normal(seed=0)))
    # model_fe.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    # convbase.trainable = False 
    # model_fe.compile(optimizer=optimizers.RMSprop(1e-5),loss='binary_crossentropy',metrics=['acc'])
    # print(len(model_fe.trainable_weights))
    # model_with_augmentation(train_dir,val_dir,model_fe)

    # fune tuning 
    tag = False 
    for layer in convbase.layers:
        if layer.name == 'block5_conv1':
            tag = True 
        if tag:
            layer.trainable = True 
        else:
            layer.trainable = False
    model_finetuning = models.Sequential()
    model_finetuning.add(convbase)
    model_finetuning.add(layers.Flatten())
    model_finetuning.add(layers.Dense(256,activation='relu',kernel_initializer=initializers.glorot_normal(seed=0)))
    model_finetuning.add(layers.Dense(1,activation='sigmoid',kernel_initializer=initializers.glorot_normal(seed=0)))
    model_finetuning.compile(optimizer=optimizers.RMSprop(lr=1e-5),loss='binary_crossentropy',metrics=['acc'])
    print(len(model_finetuning.trainable_weights))
    model_without_augmentation(train_dir,val_dir,model_finetuning)
    







