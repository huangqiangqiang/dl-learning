import numpy as np
import tensorflow as tf;
from tensorflow.keras.datasets import imdb;
from tensorflow.keras import models
from tensorflow.keras import layers


def createModel():
  model = models.Sequential()
  model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
  model.add(layers.Dense(16, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  return model

def train():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(train_data.shape)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    # print(decoded_review)

    model = createModel()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # reverse_word_index = dict()
    # word_index = imdb.get_word_index()
    # for (key, value) in word_index.items():
    #     reverse_word_index[value] = key

    # print(reverse_word_index)
    # print('---------------------------------------------')
    # print(train_data[0])
    # print('---------------------------------------------')
    # decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results
    

# docker run -it --rm --name tensorflow-cpu -v $(pwd):/app tensorflow/tensorflow:2.4.1 bash
if __name__ == '__main__':
  print(tf.__version__)

  train()

