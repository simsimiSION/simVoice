#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: triplet_model.py

from base.res_block import res_block_v1
from base.lstm_block import res_lstm_block
from base.loss import triplet_loss
from tools.data_utils import load_format_pickle, sample_voices, selcet_triplet
from dataset.voiceset import aishell_voice

import itertools
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


class triplet_model(object):
    """
    triplet 方式构建模型
    """
    def __init__(self,  mode):
        self.pickle_path = '../resource/aishell_triplet_mfcc.pickle'
        self.save_path = '../resource/model_param_best/simi_model.ckpt'
        self.top_1_path = '../resource/top_1_curve.npy'
        self.top_3_path = '../resource/top_3_curve.npy'
        self.top_1_curve = []
        self.top_3_curve = []

        self.MFCC_STRIDE = 256  # 要跟数据集保持一致
        self.MFCC_FEATURE = 20
        self.RES_CHANNEL = 64
        self.EMBEDDING_SIZE = 128

        self.BATCH_SIZE = 30
        self.EPOCH_SIZE = 1000
        self.INIT_LEARNING_RATE = 5e-3
        self.LEARNING_RATE_DECAY_EPOCH = 100
        self.LEARNING_RATE_DECAY_FACTOR = 1.0

        self.ALPHA = 0.2
        self.PEOPLE_PRE_BATCH = 20
        self.VOICES_PRE_BATCH = 60

        self.LSTM_LAYER_NUM = 2
        self.LSTM_HIDDEN_NODES = 512
        self.LSTM_DENSE_NUM = 512

        self.mode = mode
        self.voice = self.load_dataset()

        self.keep_prob = tf.placeholder(tf.float32, [])
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_inc = self.global_step.assign_add(1)


    def build_model(self, input):
        # 1. cnn
        x = res_block_v1(input)
        # 2. rnn
        output= res_lstm_block(x, self.LSTM_LAYER_NUM, self.LSTM_HIDDEN_NODES, self.BATCH_SIZE, keep_prob=1.0, reuse=False, scope='res_lstm')
        h_state = output[:,-1,:]
        # 3. dense
        dense_output = slim.fully_connected(h_state, self.EMBEDDING_SIZE, scope='feature_dense')
        embeddings = tf.nn.l2_normalize(dense_output, 1, 1e-10, name='embeddings')

        # 获取三元
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, self.EMBEDDING_SIZE]), 3, 1)
        # 计算loss
        triplets_loss = triplet_loss(anchor, positive, negative, self.ALPHA)
        # learning_rate
        learning_rate = tf.train.exponential_decay(self.INIT_LEARNING_RATE,
                                                   self.global_step,
                                                   self.LEARNING_RATE_DECAY_EPOCH * self.EPOCH_SIZE,
                                                   self.LEARNING_RATE_DECAY_FACTOR,
                                                   staircase=True)
        # 获得可优化参数,并限制权重变化
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(triplets_loss, tvars), 5)

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        return embeddings, triplets_loss, train_op


    def train(self):
        epoch = 0

        # TODO 数据构建
        input = tf.placeholder(tf.float32, (None, self.MFCC_STRIDE, self.MFCC_FEATURE, 1))
        embeddings, triplets_loss, train_op = self.build_model(input)

        # 参数初始化
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # 初始化
            sess.run(init)
            saver = tf.train.Saver()

            while epoch < self.EPOCH_SIZE:

                # ---------------------------------------------------------------#
                #  step 1. 首先进行一次前向传播,获得向量参数
                # ---------------------------------------------------------------#
                # 数据采样
                voice_arrays, sampled_class_indices, num_pre_class = sample_voices(self.voice, self.PEOPLE_PRE_BATCH, self.VOICES_PRE_BATCH)

                # 样本数
                nrof_examples = self.PEOPLE_PRE_BATCH * self.VOICES_PRE_BATCH
                # label数据( reshape成(none,3) )
                labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
                # embedding 数组
                emb_array = np.zeros((nrof_examples, self.EMBEDDING_SIZE))
                nrof_batches = int(np.ceil(nrof_examples / self.BATCH_SIZE))

                # 进行一次前向传播,获得相应的向量数据
                for i in range(nrof_batches):
                    batch_size = min(nrof_examples-i*self.BATCH_SIZE, self.BATCH_SIZE)
                    emb = sess.run(embeddings, feed_dict={input:voice_arrays[i*batch_size:(i+1)*batch_size]})
                    emb_array[i*batch_size:(i+1)*batch_size, :] = emb

                # 三元数组抽样
                triplet, nrof_random_negs, nrof_triplets = selcet_triplet(emb_array,
                                                                          num_pre_class,
                                                                          voice_arrays,
                                                                          self.PEOPLE_PRE_BATCH,
                                                                          self.ALPHA)

                triplet = list(itertools.chain(*triplet))
                # ---------------------------------------------------------------#
                #  step 2. 通过获得的向量参数,进行结构优化
                # ---------------------------------------------------------------#
                nrof_batches = int(np.ceil(nrof_triplets*3 / self.BATCH_SIZE))
                nrof_examples = len(triplet)
                i = 0
                while i < nrof_batches:
                    batch_size = min(nrof_examples-i*self.BATCH_SIZE, self.BATCH_SIZE)

                    error, _, step, emb = sess.run([triplets_loss, train_op, self.global_step_inc, embeddings],
                                                feed_dict={input:triplet[i*batch_size:(i+1)*batch_size]})

                    print('epoch:{} step:{} loss:{} '.format(epoch, i, error))
                    # ^-^
                    i += 1


                # ^-^
                saver.save(sess, self.save_path)
                epoch += 1

    def load_dataset(self):
        return load_format_pickle(self.pickle_path)




if __name__ == '__main__':

    simsimi = triplet_model( None)
    simsimi.train()
