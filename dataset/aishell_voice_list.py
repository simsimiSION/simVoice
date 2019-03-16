#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: voice_list.py


import pandas as pd
import os


# 文件地址
FILE_PATH = '/home/simsimi/文档/voiceset/voiceset_train4'
TEMP_PATH = 'train'


def make_csv():
    labels = []
    voice_path = []

    id_list = os.listdir(FILE_PATH)
    for index, id_path in enumerate(id_list):
        # 设置文件路径
        temp_path = os.path.join(FILE_PATH, id_path)
        temp_path = os.path.join(temp_path, TEMP_PATH)
        temp_path = os.path.join(temp_path, id_path)
        # 查看文件名
        voice_list = os.listdir(temp_path)
        voice_list = [os.path.join(temp_path, voice) for voice in voice_list]

        label_list = [index] * len(voice_list)
        # 设置标签
        labels.extend(label_list)
        voice_path.extend(voice_list)

    content = {}
    content['label'] = labels
    content['path'] = voice_path

    voice_list = pd.DataFrame(content)
    voice_list.to_csv('../resource/aishell_voice_4.csv')


if __name__ == "__main__":
    make_csv()

