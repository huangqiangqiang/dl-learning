import numpy as np
import tensorflow as tf;
from tensorflow.keras.datasets import imdb;
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt


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

    x_val = x_train[:10000]
    y_val = y_train[:10000]
    partial_x_train = x_train[10000:]
    partial_y_train = y_train[10000:]
    history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))

    print(history.history)

    results  = model.evaluate(x_test, y_test)

    print(results)

    res = model.predict(x_test)

    print(res)
    
    # show graph
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
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

