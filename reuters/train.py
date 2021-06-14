import numpy as np
import tensorflow as tf;
from tensorflow.keras.datasets import reuters;
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def createModel():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    return model

def train():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    decoded_review = decode(train_data[10])
    print('-------------------------------- decoded_review', decoded_review)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    one_hot_train_labels = to_one_hot(train_labels)
    one_hot_test_labels = to_one_hot(test_labels)

    model = createModel()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    x_val = x_train[:1000]
    y_val = one_hot_train_labels[:1000]
    partial_x_train = x_train[1000:]
    partial_y_train = one_hot_train_labels[1000:]

    history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))
    # print(history.history)

    result = model.evaluate(x_test, one_hot_test_labels)
    print(result)

    predictions = model.predict(x_test)
    print(np.sum(predictions[0]), np.argmax(predictions[0]))
    

    # show graph
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, acc, 'bo', label='training loss')
    plt.plot(epochs, val_acc, 'b', label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def decode(data):
    reverse_word_index = dict()
    word_index = reuters.get_word_index()
    for (key, value) in word_index.items():
        reverse_word_index[value] = key
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data])
    return decoded_review

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

def to_one_hot(labels, dimension=46):
  results = np.zeros((len(labels), dimension))
  for i, label in enumerate(labels):
    results[i, label] = 1.
  return results

# docker run -it --rm --name tensorflow-cpu -v $(pwd):/app tensorflow/tensorflow:2.4.1 bash
if __name__ == '__main__':
  print(tf.__version__)

  train()

