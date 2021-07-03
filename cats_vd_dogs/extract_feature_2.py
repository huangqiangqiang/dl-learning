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

  print(history.history)

# 画图
def show():
  loss = [0.6916903257369995, 0.5809966921806335, 0.5216398239135742, 0.4727729856967926, 0.4336850047111511, 0.43542754650115967, 0.4177835285663605, 0.3935215473175049, 0.40077027678489685, 0.39283403754234314, 0.3651161193847656, 0.37313923239707947, 0.37202465534210205, 0.3577785789966583, 0.3623710572719574, 0.34519267082214355, 0.3526455760002136, 0.3542257249355316, 0.35336634516716003, 0.3363925814628601, 0.33223068714141846, 0.34049829840660095, 0.3324885368347168, 0.33582544326782227, 0.3332201838493347, 0.3363853991031647, 0.3282449245452881, 0.3266439437866211, 0.3288900554180145, 0.3191562294960022]
  acc = [0.5954999923706055, 0.6890000104904175, 0.7294999957084656, 0.7699999809265137, 0.8059999942779541, 0.7914999723434448, 0.8065000176429749, 0.8199999928474426, 0.8144999742507935, 0.8159999847412109, 0.8385000228881836, 0.8364999890327454, 0.8295000195503235, 0.8475000262260437, 0.8429999947547913, 0.8454999923706055, 0.8374999761581421, 0.8410000205039978, 0.8429999947547913, 0.8485000133514404, 0.8554999828338623, 0.8450000286102295, 0.8560000061988831, 0.859000027179718, 0.8544999957084656, 0.8475000262260437, 0.8519999980926514, 0.8579999804496765, 0.8500000238418579, 0.8565000295639038]
  val_loss = [0.5019174814224243, 0.4238172769546509, 0.3767566978931427, 0.35077226161956787, 0.3267747163772583, 0.3131948709487915, 0.3042549192905426, 0.29728004336357117, 0.28855016827583313, 0.28379616141319275, 0.28656214475631714, 0.27057719230651855, 0.270133376121521, 0.2642974257469177, 0.26462122797966003, 0.26062554121017456, 0.2608834505081177, 0.2556924819946289, 0.25338244438171387, 0.25294917821884155, 0.2522677183151245, 0.25781765580177307, 0.2501024007797241, 0.2494092881679535, 0.24807919561862946, 0.2473958283662796, 0.24652360379695892, 0.24489614367485046, 0.24640728533267975, 0.24311600625514984]
  val_acc = [0.8100000023841858, 0.8389999866485596, 0.8510000109672546, 0.8579999804496765, 0.8669999837875366, 0.8759999871253967, 0.8690000176429749, 0.8840000033378601, 0.8799999952316284, 0.8889999985694885, 0.871999979019165, 0.8889999985694885, 0.8930000066757202, 0.8949999809265137, 0.8889999985694885, 0.8960000276565552, 0.8939999938011169, 0.8960000276565552, 0.8980000019073486, 0.8960000276565552, 0.8980000019073486, 0.8880000114440918, 0.8999999761581421, 0.8989999890327454, 0.8960000276565552, 0.8960000276565552, 0.8980000019073486, 0.8980000019073486, 0.8960000276565552, 0.906000018119812]

  epochs = range(1, len(acc) + 1)

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
  # train()
  show()