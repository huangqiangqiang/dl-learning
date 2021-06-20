import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing import image
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

def load_data():
  """
  生成训练数据和
  """
  batch_size = 100

  # 输入目录的路径，并生成批量的增强/标准化的数据。
  train_gen = image.ImageDataGenerator(
    rescale=1.0/255, # 标准化数据，默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）
    # rotation_range=20, # 随机旋转的度数范围
    # width_shift_range=0.2, # 进行水平位移
    # height_shift_range=0.2, # 进行垂直位移
    # shear_range=0, # 剪切强度（以弧度逆时针方向剪切角度）
    # zoom_range=0.5, # 随机缩放范围
    # horizontal_flip=False, # 随机水平翻转
    )
  # 训练数据
  train_flow = train_gen.flow_from_directory(directory='./data/train', color_mode='grayscale', target_size=(28,28), batch_size=batch_size, class_mode='categorical')

  # 验证数据
  val_gen = image.ImageDataGenerator(rescale=1.0/255)
  val_flow = val_gen.flow_from_directory(directory='./data/val', color_mode='grayscale', target_size=(28,28), batch_size=batch_size, class_mode='categorical')
  return train_flow, val_flow

def createModel():
  """
  创建用于深度学习的神经网络

  """
  # 搭建网络
  model = Sequential()

  # 第一层
  model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  # 第四层
  model.add(Flatten())

  # 输出层
  model.add(Dense(64, activation='relu'))
  model.add(Dense(10, activation='softmax'))

  return model

def predict(img_path):
  im = Image.open(img_path)
  resizedIm = np.array(im)
  resizedIm = resizedIm.reshape((28, 28, 1))
  resizedIm = resizedIm.astype('float32') / 255
  # print(resizedIm)
  # plt.imshow(resizedIm)
  # plt.show()
  
  dict = OrderedDict()
  dict[0] = resizedIm
  li = list(dict.values())
  li = np.array(li)

  model = createModel()
  model.load_weights("weight.h5")
  pred = model.predict(li)
  print(pred.argmax())
  # if pred.argmax() == 0:
  #   print("0")
  # else:
  #   print("其他")

def train():
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


  # plt.imshow(train_images[0])
  # plt.show()
  
  train_images = train_images.reshape((60000, 28, 28, 1))
  train_images = train_images.astype('float32') / 255

  test_images = test_images.reshape((10000, 28, 28, 1))
  test_images = test_images.astype('float32') / 255
  # 转成 one-hot 形式
  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)

  # 创建 model
  model = createModel()

  model.summary()

  # print(train_images[0])

  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_images, train_labels, epochs=5, batch_size=64)

  # test_loss, test_acc = model.evaluate(test_images, test_labels)
  # print(test_acc)

  model.save_weights('./weight.h5')

if __name__ == '__main__':
  print(tf.__version__)

  # 预测
  # for i in range(100):
  #   predict('./data/val/1/1.'+str(i+4000)+'.jpg')

  # 训练
  train()