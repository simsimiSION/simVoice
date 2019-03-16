#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: data_utils.py

from tqdm import tqdm
from pprint import pprint

import pandas as pd
import numpy as np
import librosa
import pickle


# ----------------------csv 文件处理--------------------------#
# 整合csv文件
def csv_concat(csv_path_list, save_path):
    if len(csv_path_list) == 0:
        raise Exception(' csv地址列表不能为空 ')
    else:
        for csv_path_inner in csv_path_list:
            csv = pd.read_csv(csv_path_inner, index_col=None)
            csv.to_csv(save_path, index=False,  mode='a+',)

# csv文件转化为pickle文件
def csv_to_pickle(csv_path, pickle_path):
    csv = pd.read_csv(csv_path).values

    labels = csv[:,0]
    labels = [int(i) for i in labels]
    paths = csv[:,1]

    for path, label in tqdm(zip(voice_path, labels)):
        complete, mfcc = self.voice_mfcc(path)
        if complete == False:
            continue
        elif isinstance(label, int):
            aishell_mfcc.append(mfcc)
            aishell_label.append(label)
            aishell_indices.append(index)
            index += 1

    with open(pickle_path, 'wb') as mfcc_pickle:
        pickle.dump(aishell_mfcc, mfcc_pickle)
        pickle.dump(aishell_label, mfcc_pickle)
        pickle.dump(aishell_indices, mfcc_pickle)

# ----------------------pickle 文件处理--------------------------#
# 读取保存的pickle文件
def load_mfcc_pickle(pickle_path):
    with open(pickle_path, 'rb') as mfcc_pickle:
        aishell_mfcc = pickle.load(mfcc_pickle)
        aishell_label = pickle.load(mfcc_pickle)
        aishell_indices = pickle.load(mfcc_pickle)

    return aishell_mfcc, aishell_label, aishell_indices

# 读取format pickle文件
def load_format_pickle(pickle_path):
    with open(pickle_path, 'rb') as mfcc_pickle:
        aishell = pickle.load(mfcc_pickle)

    return aishell

# pickle format
def pickle_format(pickle_path, save_path):
    aishell_mfcc, aishell_label, aishell_indices = load_mfcc_pickle(pickle_path)

    indice_dict = {}
    for mfcc, label in tqdm(zip(aishell_mfcc, aishell_label)):
        if label not in indice_dict.keys():
            indice_dict[label] = [mfcc]
        else:
            indice_dict[label].append(mfcc)

    with open(save_path, 'wb') as mfcc_pickle:
        pickle.dump(indice_dict, mfcc_pickle)

# ----------------------dataset 处理--------------------------#
# 数据采样
def sample_voices(dataset, people_pre_batch, voices_pre_person):
    """
     获得的mfcc数据未经过prehandle
    :param dataset:
    :param people_pre_batch:
    :param voices_pre_person:
    :return:
    """
    nrof_voices = people_pre_batch * voices_pre_person

    # 对类进行采样
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    voices_arrays = []
    num_pre_class = []
    sampled_class_indices = []

    # 音频采样
    while len(voices_arrays) < nrof_voices:
        # 获取文件索引
        class_index = class_indices[i]
        # 获取类中的音频数量,索引
        nrof_voices_in_class = len(dataset[class_index])
        voices_indices = np.arange(nrof_voices_in_class)
        np.random.shuffle(voices_indices)
        # 获取数据索引
        nrof_voices_from_class = min(nrof_voices_in_class,
                                     voices_pre_person,
                                     nrof_voices-len(voices_arrays))
        idx = voices_indices[:nrof_voices_from_class]
        # 获取batch音频
        voices_arrays_for_class = [mfcc_prehandle(dataset[class_index][j]) for j in idx]

        # 获取label
        sampled_class_indices += [class_index] * nrof_voices_from_class
        # 获取mfcc
        voices_arrays += voices_arrays_for_class
        num_pre_class.append(nrof_voices_from_class)

        # ^-^
        i += 1

    return voices_arrays, sampled_class_indices, num_pre_class


def selcet_triplet(embeddings, nrof_voices_per_class, voice_arrays, people_per_batch, alpha):
    """
    Select the triplets for training
    :param embeddings:
    :param nrof_voices_per_class:
    :param voice_arrays:
    :param people_per_batch:
    :param alpha:
    :return:
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in range(people_per_batch):
        # 获取当前类的音频个数
        nrof_voices = int(nrof_voices_per_class[i])
        for j in range(1, nrof_voices):
            a_idx = emb_start_idx + j - 1
            # 距离
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx]-embeddings), 1)
            for pair in range(j, nrof_voices):
                p_idx = emb_start_idx + pair
                # 正向距离
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_voices] = np.NaN
                # 负类的索引
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0]
                nrof_random_negs = all_neg.shape[0]

                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((voice_arrays[a_idx], voice_arrays[p_idx], voice_arrays[n_idx]))

                    trip_idx += 1
                num_trips += 1
        emb_start_idx += nrof_voices

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)



# ----------------------mfcc 处理--------------------------#
# 读取音频文件并通过mfcc转化
def voice_to_mfcc(wav):
    try:
        wav, sr = librosa.load(wav, mono=True)
        mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
        return True, mfcc
    except:
        print(str(wav) + ' 加载错误')
        return False, None

# mfcc 预处理
def mfcc_prehandle(mfcc, MFCC_STRIDE=256):
    if isinstance(mfcc, np.ndarray):
        length = mfcc.shape[0]
    else:
        raise Exception('mfcc文件格式错误')

    if length > MFCC_STRIDE:
        random_scope = length - MFCC_STRIDE
        random_index = np.random.randint(0, random_scope)

        pre_data = mfcc[random_index:random_index + MFCC_STRIDE, :]
    else:
        misszero_number = MFCC_STRIDE - length
        pre_data = np.zeros((MFCC_STRIDE, mfcc.shape[1]))
        pre_data[misszero_number:, :] = mfcc
    pre_data = np.expand_dims(pre_data, -1)

    return pre_data


if __name__ == '__main__':
    pickle_path = '../resource/aishell_mfcc.pickle'
    save_path = '../resource/aishell_triplet_mfcc.pickle'

    aishell = load_format_pickle(save_path)
    a, b, c = sample_voices(aishell, 10, 20)


    embeddings = np.random()

    triplets, num_trips, triplets_len = selcet_triplet_with_label(embeddings, b, c, a, 10, 0.2)





