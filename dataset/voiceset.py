#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: voiceset.py

from pprint import pprint
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import librosa
import os

CSV_PATH = '/home/simsimi/桌面/dl_code/simVoice/resource/aishell_voice_1.csv'
PICKLE_PATH = '/home/simsimi/桌面/dl_code/simVoice/resource/aishell_mfcc_test.pickle'




#TODO 数据集合并
class aishell_voice(object):
    """
    AISHELL语音处理数据集
    """
    def __init__(self, train_ratio=0.7):
        SEED = 1225
        self.MFCC_STRIDE = 256
        self.train_ratio = train_ratio

        self.TRAIN = 0
        self.TEST = 1
        self.index = 0
        self.data = []
        self.id = []

        self.aishell_mfcc, self.aishell_label, self.aishell_indices = self.load_mfcc_pickle()
        np.random.seed(SEED)
        np.random.shuffle(self.aishell_indices)
        self.length = len(self.aishell_label)
        self.id_length = np.array(self.aishell_label).max() + 1

        # 获得训练集和测试集
        self.aishell_indices_train = self.aishell_indices[:int(self.length*self.train_ratio)]
        self.aishell_indices_test = self.aishell_indices[int(self.length*self.train_ratio):]
        self.train_length = len(self.aishell_indices_train)
        self.test_length = len(self.aishell_indices_test)

    # ^-^
    def next(self, mode,batch_size=32):
        epoch = False

        # 防止溢出
        if mode == self.TRAIN:
            length = self.train_length
        elif mode == self.TEST:
            length = self.test_length
        else:
            raise Exception('MODE格式错误')

        if self.index + batch_size >= length:
            self.index = 0
            epoch = True

        # 获得当前batch的索引
        if mode == self.TRAIN:
            indices = self.aishell_indices_train[self.index: self.index + batch_size]
        elif mode == self.TEST:
            indices = self.aishell_indices_test[self.index: self.index + batch_size]

        # mfcc, id
        batch_wavs = [self.data_pretreat(self.aishell_mfcc[i]) for i in indices]
        batch_labels = [self.id_2_label(self.aishell_label[i]) for i in indices]

        # index自加
        self.index += batch_size
        return batch_wavs, batch_labels, epoch

    # 讲mfcc转化后的矩阵转化成pickle文件
    def save_mfcc_pickle(self):
        index = 0
        aishell_mfcc = []
        aishell_label = []
        aishell_indices = []

        # label, 音频地址
        csv_content = pd.read_csv(CSV_PATH).values
        labels      = csv_content[:, 1]
        voice_path = csv_content[:, 2]

        for path, label in tqdm(zip(voice_path, labels)):
            complete, mfcc = self.voice_mfcc(path)
            if complete == False:
                continue
            elif isinstance(label, int):
                aishell_mfcc.append(mfcc)
                aishell_label.append(label)
                aishell_indices.append(index)
                index += 1

        with open(PICKLE_PATH, 'wb') as mfcc_pickle:
            pickle.dump(aishell_mfcc, mfcc_pickle)
            pickle.dump(aishell_label, mfcc_pickle)
            pickle.dump(aishell_indices, mfcc_pickle)

    # 读取保存的pickle文件
    def load_mfcc_pickle(self):
        with open(PICKLE_PATH, 'rb') as mfcc_pickle:
            aishell_mfcc = pickle.load(mfcc_pickle)
            aishell_label = pickle.load(mfcc_pickle)
            aishell_indices = pickle.load(mfcc_pickle)

        return aishell_mfcc, aishell_label, aishell_indices

    # 读取音频文件并通过mfcc转化
    def voice_mfcc(self, wav):
        try:
            wav, sr = librosa.load(wav, mono=True)
            mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
            return True, mfcc
        except:
            print(str(wav) + ' 加载错误')
            return False, None

    # 数据缺省值处理
    def data_pretreat(self, data):
        if isinstance(data, np.ndarray):
            length = data.shape[0]
        else:
            raise Exception('data文件格式错误')

        if length > self.MFCC_STRIDE:
            random_scope = length - self.MFCC_STRIDE
            random_index = np.random.randint(0, random_scope)

            pre_data = data[random_index:random_index + self.MFCC_STRIDE, :]
        else:
            misszero_number = self.MFCC_STRIDE - length
            pre_data = np.zeros((self.MFCC_STRIDE, data.shape[1]))
            pre_data[misszero_number:, :] = data
        pre_data = np.expand_dims(pre_data, -1)

        return pre_data

    # 将id转化成对应的label
    def id_2_label(self, id):
        label = np.zeros((self.id_length))
        label[id] = 1
        return label


if __name__ == '__main__':
    aishell = aishell_voice()
    while True:
        a, b, c = aishell.next(aishell.TEST)
        if c == True:
            print('haha')

    print(aishell.id_length)
    print(len(a))
    print(a[0].shape)
 #   print(b)
#    print(a[0])

