import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

# docker run -it -v $(pwd)/mnist:/app hqqsk8/tensorflow:2.4.1 bash

def createModel():
  model = Sequential()

  # 第一层
  model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

  # 第四层
  model.add(Flatten())

  # # 输出层
  model.add(Dense(64, activation='relu'))
  model.add(Dense(10, activation='softmax'))

  model.summary()
  return model

def train():
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # train_images.shape = (60000, 28, 28)
  # test_images.shape = (10000, 28, 28)
  train_images = train_images.reshape((60000, 28, 28, 1))
  train_images = train_images.astype('float32') / 255
  print(test_images.shape)
  test_images = test_images.reshape((10000, 28, 28, 1))
  test_images = test_images.astype('float32') / 255

  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)


  model = createModel()
  model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_images, train_labels, epochs=5, batch_size=64)

  results = model.evaluate(test_images, test_labels)
  print(results)
# docker run -it --name tensorflow-gpu -v $(pwd)/mnist:/app hqqsk8/tensorflow:2.4.1-gpu-jupyter bash
if __name__ == '__main__':
  print(tf.__version__)

  # 训练
  train()