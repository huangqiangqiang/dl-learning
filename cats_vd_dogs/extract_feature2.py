from scipy.ndimage.measurements import label
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

"""特征提取
使用训练好的神经网络，提取他们的卷积层，或者是卷积层的前几层，因为最开始的几层提取的是纹理等特征，通用型较广。
然后自定义全连接层，全连接层就是用来判断最后的分类。

这里使用的是提取 VGG16 模型的卷积层，去掉后面的全连接层。
因为 VGG16 模型能识别上千种分类，最后的输出层也是有上千个神经元，虽然包含了不同种类的猫和狗，但是我们想要的只是能识别猫狗就行了。
所以我们自定义了全连接层和输出层。
"""

conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

base_dir = '../../dataset/cats_and_dogs_small'
train_dir = base_dir + '/train'
val_dir = base_dir + '/validation'
test_dir = base_dir + '/test'

def train():
  # 使用数据增强
  train_datagen = image.ImageDataGenerator(
    rescale=1./255, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' 
  )
  val_datagen = image.ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
  val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

  # 冻结 conv_base 的权重，训练时不会改变，需要在模型编译前设置
  conv_base.trainable = False

  model = Sequential()
  model.add(conv_base)
  model.add(Flatten())
  model.add(Dense(256, activation='relu', input_dim=4 * 4 * 512))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.summary()
  model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='binary_crossentropy', metrics=['acc'])

  history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=val_generator, validation_steps=50)

  # 画图
  acc = history.history['acc']
  loss = history.history['loss']
  val_acc = history.history['val_acc']
  val_loss = history.history['val_loss']
  epochs = (1, 31)

  plt.plot(epochs, acc, 'bo', label='train acc')
  plt.plot(epochs, val_acc, 'b', label='validation acc')
  plt.title('training and validation acc')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='train loss')
  plt.plot(epochs, val_loss, 'b', label='validation loss')
  plt.title('training and validation loss')
  plt.legend()

  plt.show()


if __name__ == '__main__':
  print(tf.__version__)

  # 训练
  train()