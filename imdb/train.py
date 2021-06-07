import tensorflow as tf;
from tensorflow.keras.datasets import imdb;

def train():
    # 仅仅返回数据中最常出现的前 10000 个单词
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(train_data.shape)
    print( train_data[0])
    reverse_word_index = dict()
    word_index = imdb.get_word_index()
    for (key, value) in word_index.items():
        reverse_word_index[value] = key
    print(reverse_word_index)


if __name__ == '__main__':
  print(tf.__version__)

  # 训练
  train()