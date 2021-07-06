from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing import image

class CatDogClassifier:
  """猫狗分类器
  """

  def load_data(self, train_directory):
    train_datagen = image.ImageDataGenerator(rescale=1./255)
    val_datagen = image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_directory + '/train', target_size=(150, 150), batch_size=32)
    val_generator = val_datagen.flow_from_directory(train_directory + '/val', target_size=(150, 150), batch_size=32)

    return train_generator, val_generator


  def createModel(self):
    model = Sequential()
    model.add(Conv2D(filters=32 , kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64 , kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128 , kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128 , kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summery()
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
    self.history = model.fit_generator(train_generator, epochs=30, validation_data=val_generator)

  def showHistory(self):
    print(1)

  def predict(self, predict_image_path):
    """预测，根据图片地址预测类别.

    Args:
        predict_image_path: 预测的图片路径
    """



if __name__ == '__main__':
  classifier = CatDogClassifier()
  classifier.train('../../dataset/cat_vs_dog')