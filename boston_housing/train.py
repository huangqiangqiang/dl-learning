import numpy as np
import tensorflow as tf;
from tensorflow.keras.datasets import boston_housing;
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def createModel():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # mse：【mean squared error】 为均方误差损失函数，预测值与真实值之间的误差，回归问题常用的损失函数
    # mae：【mean absolute error】预测值与真实值之差的绝对值
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def train():
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    print('--------------------------------', train_data.shape, train_data[0], train_targets[0])

    # 不同的特征取值范围差异很大，直接输入到神经网络中是有问题的，需要对每个特征做标准化，让他们映射到 -1～1之间的值
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    print('--------------------------------', train_data[0])

    k = 4
    # // 表示取整除，取整数部分
    num_val_samples = len(train_data) // k
    all_scores = []
    all_mae_history = []
    num_epochs = 500
    for i in range(k):
      print('processing fold #', i)
      val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
      val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

      partial_train_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]
      ], axis=0)
      partial_train_targets = np.concatenate([
        train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]
      ], axis=0)

      model = createModel()
      history = model.fit(partial_train_data, partial_train_targets, batch_size=1, epochs=num_epochs, validation_data=(val_data, val_targets))
      print(history.history.keys()) 
      mae_history = history.history['val_mae']
      val_mse, val_mae = model.evaluate(val_data, val_targets)
      all_scores.append(val_mae)
      all_mae_history.append(mae_history)
    
    print('=================================')
    print(all_scores, np.mean(all_scores))

    average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
    print(average_mae_history)

    # show graph
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('epochs')
    plt.ylabel('validation mae')
    plt.legend() # 给图加上图例
    plt.show()

# docker run -it --rm --name tensorflow-cpu -v $(pwd):/app tensorflow/tensorflow:2.4.1 bash
if __name__ == '__main__':
  print(tf.__version__)

  train()

