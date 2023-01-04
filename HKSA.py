import json
import logging
import random
from typing import Any

import jieba
import numpy as np
from keras import Sequential
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing import text
from keras.models import model_from_json
from keras.utils import pad_sequences


def words2dict(words):
    tokenizer = text.Tokenizer()
    word_dic = {}
    voca = []
    for sentence in words:
        cut = list(jieba.cut(sentence))
        cut.append('unk')
        for word in cut:
            if not (word in word_dic):
                word_dic[word] = 0
            else:
                word_dic[word] += 1
            voca.append(word)
    word_dic = sorted(word_dic.items(), key=lambda kv: kv[1], reverse=True)
    voca = [v[0] for v in word_dic]
    tokenizer.fit_on_texts(voca)
    return len(voca), tokenizer.word_index


def words2index(words_dict, words):
    rt_list = []
    for j in words:
        words_list = list(jieba.cut(j, cut_all=False))
        index_list = []
        for i in words_list:
            if i in words_dict:
                index_list.append(words_dict[i])
            else:
                index_list.append(words_dict['unk'])

        rt_list.append(np.array(index_list))
    return np.array(rt_list, dtype=object)


def model_lstm(cuda=False, output_dim=2, max_len=10, dict_len=10):
    if cuda:
        lstm = CuDNNLSTM
    else:
        lstm = LSTM
    model = Sequential()
    model.add(Embedding(trainable=True, input_dim=dict_len,
                        output_dim=150, input_length=max_len))
    model.add(Bidirectional(lstm(32, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.6))
    model.add(Dense(96, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Bidirectional(lstm(24), merge_mode='concat'))
    model.add(Dropout(0.6))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def model_conv(output_dim=2, max_len=10, dict_len=10):
    model = Sequential()
    model.add(Embedding(trainable=True, input_dim=dict_len,
                        output_dim=512, input_length=max_len))

    model.add(Conv1D(kernel_size=5, filters=100))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(kernel_size=3, filters=100))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Dense(1024, activation='relu'))

    model.add(Flatten())
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def d_data2one(data1: str, data2: str):
    min_ = min(len(data1), len(data2))
    rg = random.randint(0, min_)
    return data1[:rg] + data2[rg:]


class HKSA:
    def __init__(self):
        self.voca_len = None
        self.max_len = None
        self.keys = []
        self.word_dict = None
        self.train_data = None
        self.model = model_lstm()

    def create_from_data(self, train_data, model=model_lstm):
        self.train_data = train_data

        sentence = ''
        len_total = 0
        for i in train_data:
            sentence += i + ' '
            len_total += len(i)
            if train_data[i] not in self.keys:
                self.keys.append(train_data[i])
        avg_len = len_total / len(train_data)

        voca_len, word_dict = words2dict(sentence)
        self.voca_len = voca_len
        self.word_dict = word_dict
        self.max_len = int(avg_len)
        if 'cuda' in model.__annotations__:
            self.model = model(output_dim=len(self.keys), max_len=self.max_len, dict_len=self.voca_len, cuda=False)
        else:
            self.model = model(output_dim=len(self.keys), max_len=self.max_len, dict_len=self.voca_len)

    def train(self, train_data=None,
              batch_size: Any = None,
              epochs: int = 1,
              verbose: str = "auto",
              callbacks=None,
              validation_split: float = 0.0,
              validation_data: Any = None,
              shuffle: bool = True,
              class_weight: Any = None,
              sample_weight: Any = None,
              initial_epoch: int = 0,
              steps_per_epoch: Any = None,
              validation_steps: Any = None,
              validation_batch_size: Any = None,
              validation_freq: int = 1,
              max_queue_size: int = 10,
              workers: int = 1,
              use_multiprocessing: bool = False,
              rg: int = None):
        if train_data is None:
            train_data = self.train_data
        if rg:
            for _ in range(rg):
                key1 = random.choice(list(train_data))
                v = train_data[key1]
                key2 = random.choice(list(train_data))
                while v != train_data[key2]:
                    key2 = random.choice(list(train_data))
                train_data[d_data2one(key1, key2)] = v
            print(f'rged data size: {len(train_data)}')
        x_train = words2index(self.word_dict, train_data)
        y_train = []
        for i in train_data:
            temp_list = [0 for _ in range(len(self.keys))]
            temp_list[self.keys.index(train_data[i])] = 1
            y_train.append(temp_list)

        y_train = np.array(y_train)

        indies = [i for i in range(len(x_train))]
        random.seed(114514)
        random.shuffle(indies)

        x_train = x_train[indies]
        y_train = y_train[indies]

        x_train = pad_sequences(x_train, maxlen=self.max_len)

        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks,
                       validation_split=validation_split,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       class_weight=class_weight,
                       sample_weight=sample_weight,
                       initial_epoch=initial_epoch,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       validation_batch_size=validation_batch_size,
                       validation_freq=validation_freq,
                       max_queue_size=max_queue_size,
                       workers=workers,
                       use_multiprocessing=use_multiprocessing)

    def save(self, name):
        json.dump({'model': self.model.to_json(), 'dict': self.word_dict, 'max_len': self.max_len, 'keys': self.keys},
                  open(name, 'w', encoding='utf8'))
        self.model.save_weights(name + '.h5')

    def load(self, name):
        model_json = json.load(open(name, 'r', encoding='utf8'))

        self.model = model_from_json(model_json['model'])
        self.word_dict = model_json['dict']
        self.max_len = model_json['max_len']
        self.keys = model_json['keys']
        self.model.load_weights(name + '.h5')
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def predict(self, data):
        predict_datas = words2index(self.word_dict, data)
        predict_datas = pad_sequences(predict_datas, maxlen=self.max_len)

        return [self.keys[np.argmax(i)] for i in self.model.predict(predict_datas)]

    def evaluate(self, data):
        x = words2index(self.word_dict, data)
        y = []
        for i in data:
            temp_list = [0 for _ in range(len(self.keys))]
            temp_list[self.keys.index(data[i])] = 1
            y.append(temp_list)

        y = np.array(y)

        x = pad_sequences(x, maxlen=self.max_len)
        self.model.evaluate(x, y)


class Formatter:
    @staticmethod
    def lists(lists_text: str, line_separator='\n', column_separator=',', key_location=0, value_location=1):
        rt_dict = {}
        lines = lists_text.split(line_separator)
        for line in lines:
            try:
                rt_dict[line.split(column_separator)[key_location]] = line.split(column_separator)[value_location]
            except Exception as err:
                logging.getLogger(__name__).error(err)
        return rt_dict

    @staticmethod
    def dicts(dicts_text: dict, key_key='key', value_key='value'):
        rt_dict = {}
        for i in dicts_text:
            rt_dict[i[key_key]] = i[value_key]
        return rt_dict
