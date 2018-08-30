# -*- coding:utf-8 -*- 

import os
import shutil 
from keras import layers 
from keras import models 
from keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt 

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
        height_shift_range=0.2,rotation_range=0.2,
        zoom_range=0.2,horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,batch_size=20,target_size=(150,150),class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(val_dir,batch_size=20,target_size=(150,150),class_mode='binary')
    history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=20,validation_data=validation_generator,validation_steps=50)
    model.save('cats_dogs_samll_dataaugmentation.h5')
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

    model = build_model(input_shape=(150,150,3))
    # print(model.summary())
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    model_with_augmentation(train_dir,val_dir,model)



