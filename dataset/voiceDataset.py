#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: voiceDataset.py


from pprint import pprint
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import librosa
import os


class voiceDataset(object):
    """
    语音处理数据集
    """

    def __init__(self):
        DATA_PATH = '/home/simsimi/桌面/voicedata'
        DATA_DIR = 'voiceprint_training/data'
        LIST_DIR = 'voiceprint_training/info.csv'
        SEED = 1225
        self.MFCC_STRIDE = 256

        self.index = 0
        self.data = []
        self.id = []
        # 设置读取地址
        self.data_path = os.path.join(DATA_PATH, DATA_DIR)
        self.list_path = os.path.join(DATA_PATH, LIST_DIR)
        # 读取列表文件
        self.list_content = pd.read_csv(self.list_path).values
        # 获得列表长度
        self.length = len(self.list_content)
        # 设置索引
        self.indices = np.arange(0, self.length)

        # 获取音频地址,语音信息
        voice_address = self.list_content[self.indices, 1]
        voice_id = self.list_content[self.indices, 0]

        # 为每个用户设置相对应的id数
        self.user_dict = {}
        for inner in voice_id:
            if inner not in self.user_dict:
                self.user_dict[inner] = len(self.user_dict)
        self.id_length = len(self.user_dict)

        # 设置文件地址以及id数
        for index, address in enumerate(voice_address):
            self.data.append(self.data_path + '/' + address + '.wav')
            self.id.append(self.user_dict[voice_id[index]])

        # 随机化
        np.random.seed(SEED)
        np.random.shuffle(self.indices)

        # 保存,加载pickle文件
        # self.save_mfcc_pickle()
        self.mfcc = self.load_mfcc_pickle()

    # ^-^
    def next(self, batch_size=32):
        batch_wavs = []
        batch_labels = []
        epoch = False

        # 防止溢出
        if self.index + batch_size >= self.length:
            self.index = 0
            epoch = True

        # 获得当前batch的索引
        indices = self.indices[self.index: self.index + batch_size]
        # address, id
        data = [self.data_pretreat(self.mfcc[i]) for i in indices]
        label = [self.id_2_label(self.id[i]) for i in indices]

        # index自加
        self.index += batch_size

        return data, label, epoch

    # 将id转化成对应的label
    def id_2_label(self, id):
        label = np.zeros((self.id_length))
        label[id] = 1
        return label

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

    # 读取音频文件并通过mfcc转化
    def voice_mfcc(self, wav):
        try:
            wav, sr = librosa.load(wav, mono=True)
            mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
            return mfcc
        except:
            print(str(wav) + ' 加载错误')

    # 讲mfcc转化后的矩阵转化成pickle文件
    def save_mfcc_pickle(self):
        mfcc_dict = {}

        for index in tqdm(self.indices):
            address = self.data[index]
            mfcc = self.voice_mfcc(address)
            mfcc_dict[index] = mfcc

        with open('../resource/mfcc.pickle', 'wb') as mfcc_pickle:
            pickle.dump(mfcc_dict, mfcc_pickle)

    # 读取保存的pickle文件
    def load_mfcc_pickle(self):
        with open('../resource/mfcc.pickle', 'rb') as mfcc_pickle:
            mfcc_dict = pickle.load(mfcc_pickle)

        return mfcc_dict


if __name__ == '__main__':

    voice = voiceDataset()
    a, b, _ = voice.next()

    step = 0
    while True:
        step += 1
        print(step)
        a,b,c = voice.next(32)
        if c:
            print('finish')


