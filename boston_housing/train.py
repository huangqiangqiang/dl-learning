import numpy as np
import tensorflow as tf;
from tensorflow.keras.datasets import boston_housing;
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def createModel():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    print('--------------------------------', train_data.shape, train_data[0])

    model = createModel()

    history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))
    # print(history.history)
    

    # show graph
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, acc, 'bo', label='training loss')
    # plt.plot(epochs, val_acc, 'b', label='validation loss')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()

# docker run -it --rm --name tensorflow-cpu -v $(pwd):/app tensorflow/tensorflow:2.4.1 bash
if __name__ == '__main__':
  print(tf.__version__)

  train()

