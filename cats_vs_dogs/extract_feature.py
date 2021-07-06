from scipy.ndimage.measurements import label
import numpy as np
import tensorflow as tf
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

这个例子中，我们提取了 VGG16 的卷积层，也就是 conv_base，然后我们把训练集传入 conv_base 预测，预测的输出先展开，再做为我们自定义模型的输入。
"""

conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))

base_dir = '../../dataset/cats_and_dogs_small'
train_dir = base_dir + '/train'
val_dir = base_dir + '/validation'
test_dir = base_dir + '/test'

datagen = image.ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
  features = np.zeros(shape=(sample_count, 4, 4, 512))
  labels = np.zeros(shape=(sample_count))
  generater = datagen.flow_from_directory(directory=directory, target_size=(150, 150), batch_size=batch_size, class_mode='binary')
  i = 0
  for inputs_batch, labels_batch in generater:
    features_batch = conv_base.predict(inputs_batch)
    features[i * batch_size:(i + 1) * batch_size] = features_batch
    labels[i * batch_size:(i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
      break
  return features, labels

def train():
  print('================11111111111111122222')
  train_features, train_labels = extract_features(train_dir, 2000)
  print('================22222222222222222222')
  val_features, val_labels = extract_features(val_dir, 1000)
  print('================33333333333333333333333')
  # test_features, test_labels = extract_features(test_dir, 1000)

  # 展开
  train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
  val_features = np.reshape(val_features, (1000, 4 * 4 * 512))

  model = Sequential()
  model.add(Dense(256, activation='relu', input_dim=4 * 4 * 512))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
  history = model.fit(train_features, train_labels, epochs=30, batch_size=20, validation_data=(val_features, val_labels))
  
  # 画图
  acc = history.history['acc']
  loss = history.history['loss']
  val_acc = history.history['val_acc']
  val_loss = history.history['val_loss']
  print('====================================')
  print(acc)
  print(val_acc)
  epochs = range(1, 31)

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