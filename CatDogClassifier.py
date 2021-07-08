#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.layers.core import Flatten
import sys, getopt

class CatDogClassifier:
  """猫狗分类器

  提高训练准确度的方法：
  1. 增强数据，并添加 Dropout 层
  2. 提供验证集
  3. 迁移学习
  """

  model_filename = 'CatDogClassifier_model.h5'

  def load_data(self, train_directory):
    train_datagen = image.ImageDataGenerator(rescale=1./255)
    val_datagen = image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_directory + '/train', target_size=(150, 150), batch_size=32)
    val_generator = val_datagen.flow_from_directory(train_directory + '/val', target_size=(150, 150), batch_size=32)

    return train_generator, val_generator


  def createModel(self):
    model = Sequential()
    model.add(Conv2D(filters=32 , kernel_size=(3,3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64 , kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128 , kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128 , kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model

  def train(self, train_directory):
    """训练，根据训练目录开始训练.

    Args:
        train_directory: 训练的路径
    """
    print('start training..., train directory:', train_directory)
    train_generator, val_generator = self.load_data(train_directory)

    model = self.createModel()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit_generator(train_generator, epochs=30, validation_data=val_generator)
    model.save_weights(CatDogClassifier.model_filename)
    self.showHistory(history.history)

  def showHistory(self, history):
    print(history)

  def predict(self, predict_image_path):
    """预测，根据图片地址预测类别.

    Args:
        predict_image_path: 预测的图片路径
    """
    model = self.createModel()
    model.load_weights(CatDogClassifier.model_filename)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    pred = model.predict()
    print(pred.argmax())


def main(argv):
   inputfile = ''
   try:
      opts, args = getopt.getopt(argv, "hi:", ['image='])
   except getopt.GetoptError:
      print('CatDogClassifier.py -i <inputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('CatDogClassifier.py -i <inputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
   return inputfile

if __name__ == '__main__':
  inputfile = main(sys.argv[1:])
  classifier = CatDogClassifier()
  if inputfile:
    classifier.predict(inputfile)
  else:
    classifier.train('./dataset/cat_vs_dog')