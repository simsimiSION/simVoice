#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: model_res_bi_lstm.py


from base.res_block import res_block_v1
from base.lstm_block import *
from base.loss import softmax_loss, triplet_loss
from dataset.voiceset import aishell_voice

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

TRAIN = 0
RETRAIN = 1


class simi_model(object):
    def __init__(self, voice, mode):
        self.mode = mode

        self.save_path = '../resource/model_param_best/simi_model.ckpt'
        self.top_1_path = '../resource/top_1_curve.npy'
        self.top_3_path = '../resource/top_3_curve.npy'
        self.top_1_curve = []
        self.top_3_curve = []
        self.voice = voice

        self.MFCC_STRIDE = 256  # 要跟数据集保持一致
        self.MFCC_FEATURE = 20
        self.RES_CHANNEL = 64
        self.CLS_NUMBER = 40
        self.BATCH_SIZE = 32
        self.INIT_LEARNING_RATE = 5e-3

        self.LSTM_LAYER_NUM = 2
        self.LSTM_HIDDEN_NODES = 512
        self.LSTM_DENSE_NUM = 512

        self.x = tf.placeholder(tf.float32, (None, self.MFCC_STRIDE, self.MFCC_FEATURE, 1))
        self.y = tf.placeholder(tf.float32, [None, self.CLS_NUMBER])
        self.keep_prob = tf.placeholder(tf.float32, [])

    def build_model(self):
        # 1. cnn
        x = res_block_v1(self.x)
        # 2. rnn
        output, output1 = res_bi_lstm_block(x, self.LSTM_LAYER_NUM, self.LSTM_HIDDEN_NODES, self.BATCH_SIZE, keep_prob=1.0, reuse=False, scope='res_lstm')
        h_state = output[:,-1,:] + output1[:,-1,:]
        # 3. softmax
        if self.mode == TRAIN or self.mode == RETRAIN:
            y_predict = self.sofmax(h_state, name='predict')
        else:
            raise Exception('MODE 格式错误')
        return y_predict


    def compile(self, y_predict):
        global_step = tf.Variable(0, trainable=False)
        global_step_inc = global_step.assign_add(1)

        loss = softmax_loss(y_predict, self.y)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)

        learning_rate = tf.train.exponential_decay(self.INIT_LEARNING_RATE,
                                                   global_step=global_step,
                                                   decay_steps=2000, decay_rate=0.96)

        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return loss, train_op, global_step_inc

    def get_accuracy(self, y_predict):
        target = tf.argmax(self.y, 1)
        top_1 = tf.nn.in_top_k(y_predict, target, 1)
        top_3 = tf.nn.in_top_k(y_predict, target, 3)

        top_1_accuracy = tf.reduce_mean(tf.cast(top_1, tf.float32))
        top_3_accuracy = tf.reduce_mean(tf.cast(top_3, tf.float32))

        return top_1_accuracy, top_3_accuracy

    def train(self, load_fc=False, save_fc=False):
        # 构建网络
        y_predict = self.build_model()
        loss, train_op, global_step_op = self.compile(y_predict)
        top_1_accuracy, top_3_accuracy = self.get_accuracy(y_predict)

        # 保存模型
        if load_fc:
            saver = tf.train.Saver()
        else:
            exclude = ['predict_dense']
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
            saver = tf.train.Saver(variables_to_restore)

        # 初始化
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # 初始化
            sess.run(init)
            # 判断是否加载全连接层
            if save_fc:
                saver = tf.train.Saver()
            else:
                exclude = ['predict_dense']
                variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
                saver = tf.train.Saver(variables_to_restore)
            # 加载模型参数
            # self.load_file(sess, saver)

            epoch = 0
            step = 0
            while True:
                # 产生数据
                x, y, epoch_finish = self.voice.next(self.voice.TRAIN, self.BATCH_SIZE)
                if epoch_finish:
                    epoch += 1
                    step = 0

                # 数据输出
                if step % 20 == 0:
                    top_1_accu, top_3_accu = sess.run([top_1_accuracy, top_3_accuracy], feed_dict={self.x: x,
                                                                                                   self.y: y,
                                                                                                   self.keep_prob: 1.0})

                    train_loss = loss.eval(feed_dict={self.x: x,
                                                      self.y: y,
                                                      self.keep_prob: 1.0})
                    self.top_1_curve.append(top_1_accu)
                    self.top_3_curve.append(top_3_accu)
                    print('epoch:{} step:{} loss:{} top1_accuracy:{} top3_accuracy:{}'.format(epoch, step, train_loss, top_1_accu, top_3_accu))

                # 训练
                _, global_step = sess.run([train_op, global_step_op], feed_dict={self.x: x,
                                                                                self.y: y,
                                                                                self.keep_prob: 1.0})

                # 保存数据
                if epoch_finish:
                    saver.save(sess, self.save_path)
                    np.save(self.top_1_path, np.array(self.top_1_curve))
                    np.save(self.top_3_path, np.array(self.top_3_curve))

                # ^-^
                step += 1

    def validation(self):
        pass

    def lstm_cell(self, num_nodes):
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        return cell

    def sofmax(self, input,name=None):
        x = slim.fully_connected(input, self.CLS_NUMBER, scope=name + '_dense')
        x = tf.nn.softmax(x, name=name + '_softmax')
        return x

    def load_file(self, sess,saver):
        saver.restore(sess, self.save_path)
        try:
            self.top_1_curve = list(np.load(self.top_1_path))
            self.top_3_curve = list(np.load(self.top_3_path))
        except:
            self.top_1_curve = []
            self.top_3_curve = []

if __name__ == '__main__':
    voiceset = aishell_voice()

    simi_voice = simi_model(voiceset, mode=RETRAIN)
    simi_voice.train()







