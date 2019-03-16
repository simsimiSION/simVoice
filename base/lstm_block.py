#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: bi_lstm.py

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


# ---------------------------- CELL -------------------------- #
# lstm单元
def lstm_cell(lstm_hidden_size, keep_prob, reuse=False, name=None):
    """
    lstm with dropout
    :param lstm_hidden_size: 隐藏单元数目
    :param keep_prob:
    :param reuse:
    :param name:
    :return:
    """
    if name == None:
        raise Exception('lstm cell needs name')

    cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, reuse=reuse, name=name + '_lstm')
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


# 参差lstm单元
def res_lstm_cell(lstm_hidden_size, keep_prob, residual_fn=None, reuse=False, name=None):
    """
    residual lstm with dropout
    :param lstm_hidden_size: 隐藏单元数目
    :param keep_prob:
    :param residual_fn:
    :param reuse:
    :param name:
    :return:
    """
    if name == None:
        raise Exception('lstm cell needs name')

    cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size, reuse=reuse, name=name + '_res_lstm')
    cell = tf.nn.rnn_cell.ResidualWrapper(cell, residual_fn=residual_fn)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


# ---------------------------- BLOCK -------------------------- #
# 多层lstm
def lstm_block(input, layer_num, lstm_hidden_size, batch_size, keep_prob=1.0, reuse=False, scope=None):
    """
    构建多层lstm
    :param input: 输入tensor
    :param layer_num: lstm层数
    :param lstm_hidden_size: 隐藏单元数
    :param batch_size:
    :param keep_prob:
    :param reuse:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(lstm_hidden_size, keep_prob, reuse=reuse, name='lstm_block_' + str(num)) for num in
             range(layer_num)])

        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=input, initial_state=init_state)

    return outputs


# 双向lstm
def bi_lstm_block(input, layer_num, lstm_hidden_size, batch_size, keep_prob=1.0, reuse=False, scope=None):
    """
    构建多层双向lstm
    :param input: 输入tensor
    :param layer_num: lstm层数
    :param lstm_hidden_size: 隐藏单元数
    :param batch_size:
    :param keep_prob:
    :param reuse:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        lstm_fw = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(lstm_hidden_size, keep_prob, reuse=reuse, name='fw_bi_lstm_block_' + str(num)) for num in
             range(layer_num)])
        lstm_bw = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(lstm_hidden_size, keep_prob, reuse=reuse, name='bw_bi_lstm_block_' + str(num)) for num in
             range(layer_num)])

        init_fw = lstm_fw.zero_state(batch_size, dtype=tf.float32)
        init_bw = lstm_bw.zero_state(batch_size, dtype=tf.float32)

        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                lstm_bw_cell,
                                                                input,
                                                                initial_state_fw=init_fw,
                                                                initial_state_bw=init_bw,
                                                                scope=scope + 'bi_rnn')

        outputs_fw = outputs[0]
        outputs_bw = outputs[1]

    return outputs_fw, outputs_bw


# ---------------------------- RES BLOCK -------------------------- #
# 参差多层lstm
def res_lstm_block(input, layer_num, lstm_hidden_size, batch_size, keep_prob=1.0, residual_fn=None, reuse=False,
                   scope=None):
    """
    构建多层参差lstm
        如果层数 > 2, 则在第二层使用基于参差的lstm模型
    :param input:  输入tensor
    :param layer_num: lstm层数
    :param lstm_hidden_size: 隐藏单元数
    :param batch_size:
    :param keep_prob:
    :param residual_fn:
    :param reuse:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        if layer_num > 1:
            mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(lstm_hidden_size, keep_prob, reuse=reuse, name='res_lstm_block_1')] +
                [res_lstm_cell(lstm_hidden_size, keep_prob, residual_fn=residual_fn, reuse=reuse,
                               name='res_lstm_block_' + str(num + 1)) for num in range(layer_num - 1)])
        else:
            mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [res_lstm_cell(lstm_hidden_size, keep_prob, residual_fn=residual_fn, reuse=reuse,
                               name='res_lstm_block_' + str(num)) for num in range(layer_num)])

        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=input, initial_state=init_state)

    return outputs


# 参差双向lstm
def res_bi_lstm_block(input, layer_num, lstm_hidden_size, batch_size, keep_prob=1.0, residual_fn=None, reuse=False,
                      scope=None):
    """
    构建双向多层参差lstm
        如果层数 > 2, 则在第二层使用基于参差的lstm模型
    :param input:输入tensor
    :param layer_num: lstm层数
    :param lstm_hidden_size: 隐藏单元数
    :param batch_size:
    :param keep_prob:
    :param residual_fn:
    :param reuse:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        if layer_num > 1:
            lstm_fw = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(lstm_hidden_size, keep_prob, reuse=reuse, name='fw_res_lstm_block_1')] +
                [res_lstm_cell(lstm_hidden_size, keep_prob, residual_fn=residual_fn, reuse=reuse,
                               name='fw_res_lstm_block_' + str(num + 1)) for num in range(layer_num - 1)])
            lstm_bw = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(lstm_hidden_size, keep_prob, reuse=reuse, name='bw_res_lstm_block_1')] +
                [res_lstm_cell(lstm_hidden_size, keep_prob, residual_fn=residual_fn, reuse=reuse,
                               name='bw_res_lstm_block_' + str(num + 1)) for num in range(layer_num - 1)])
        else:
            lstm_fw = tf.nn.rnn_cell.MultiRNNCell(
                [res_lstm_cell(lstm_hidden_size, keep_prob, residual_fn=residual_fn, reuse=reuse,
                               name='fw_res_lstm_block_' + str(num)) for num in range(layer_num)])
            lstm_bw = tf.nn.rnn_cell.MultiRNNCell(
                [res_lstm_cell(lstm_hidden_size, keep_prob, residual_fn=residual_fn, reuse=reuse,
                               name='bw_res_lstm_block_' + str(num)) for num in range(layer_num)])

        init_fw = lstm_fw.zero_state(batch_size, dtype=tf.float32)
        init_bw = lstm_bw.zero_state(batch_size, dtype=tf.float32)

        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw,
                                                                lstm_bw,
                                                                input,
                                                                initial_state_fw=init_fw,
                                                                initial_state_bw=init_bw,
                                                                scope=scope + 'res_bi_rnn')

        outputs_fw = outputs[0]
        outputs_bw = outputs[1]

    return outputs_fw, outputs_bw


if __name__ == '__main__':
    pass
