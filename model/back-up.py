#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: back-up.py

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


class simi_model(object):
    def __init__(self, voice):
        tf.reset_default_graph()
        self.save_path = '/content/drive/sims_voice/simVoice/resource/simi_model.ckpt'
        self.accuracy_curve_path = '/content/drive/sims_voice/simVoice/resource/accuracy_curve.npy'
        self.accuracy_curve = []
        self.voice = voice

        self.MFCC_STRIDE = 256  # 要跟数据集保持一致
        self.MFCC_FEATURE = 20
        self.RES_CHANNEL = 64
        self.CLS_NUMBER = 300
        self.BATCH_SIZE = 32

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
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(
            [self.lstm_cell(self.LSTM_HIDDEN_NODES) for _ in range(self.LSTM_LAYER_NUM)],
            state_is_tuple=True)

        init_state = mlstm_cell.zero_state(self.BATCH_SIZE, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state)
        h_state = outputs[:, -1, :]
        # 3. softmax
        y_predict = self.sofmax(h_state, name='predict')
        return y_predict

    def compile(self, y_predict):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=(y_predict),
                                                    labels=self.y))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)

        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return loss, train_op

    def get_accuracy(self, y_predict):
        correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                      tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def train(self):
        # 构建网络
        y_predict = self.build_model()
        loss, train_op = self.compile(y_predict)
        accuracy = self.get_accuracy(y_predict)
        # 保存模型
        # exclude = ['predict_dense', 'predict_softmax']
        # variable_to_restore = slim.get_variables_to_restore(exclude=exclude)
        # saver = tf.train.Saver(variable_to_restore)
        saver = tf.train.Saver()
        # 初始化
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # saver.restore(sess, self.save_path)

            epoch = 0
            step = 0

            while True:
                # 产生数据
                x, y, epoch_finish = self.voice.next(self.BATCH_SIZE)
                if epoch_finish:
                    epoch += 1
                    step = 0

                # 数据输出
                if step % 20 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={self.x: x,
                                                                   self.y: y,
                                                                   self.keep_prob: 1.0})

                    train_loss = loss.eval(feed_dict={self.x: x,
                                                      self.y: y,
                                                      self.keep_prob: 1.0})
                    self.accuracy_curve.append(train_accuracy)
                    print('epoch:{} step:{} loss:{} accuracy:{}'.format(epoch, step, train_loss, train_accuracy))

                # 训练
                sess.run(train_op, feed_dict={self.x: x,
                                              self.y: y,
                                              self.keep_prob: 1.0})

                # 保存数据
                if epoch_finish:
                    saver.save(sess, self.save_path)
                    np.save(self.accuracy_curve_path, np.array(self.accuracy_curve))

                # ^-^
                step += 1

    def lstm_cell(self, num_nodes):
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        return cell

    def sofmax(self, input, name=None):
        x = slim.fully_connected(input, self.CLS_NUMBER, scope=name + '_dense')
        x = tf.nn.softmax(x, name=name + '_softmax')
        return x
