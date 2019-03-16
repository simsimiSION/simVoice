#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: res_block.py


import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


def res_block(input, in_channel, stride=1, down_sample=None, trainable=True, scope=None):
    """
    res单元
    :param input:  输入tensor
    :param in_channel: 期望的输入channel 对应输出channel为 4*in_channel
    :param stride:
    :param down_sample: 是否进行下采样(对应tensor shape不同时)
    :param trainable:
    :param scope:
    :return: 输出tensor
    """
    with tf.name_scope(scope):

        # bottleneck
        net = slim.conv2d(input, in_channel, [1, 1], stride=stride,
                          normalizer_fn=slim.batch_norm,
                          activation_fn=tf.nn.relu,
                          trainable=trainable)
        net = slim.conv2d(net, in_channel, [3, 3],
                          normalizer_fn=slim.batch_norm,
                          activation_fn=tf.nn.relu,
                          trainable=trainable)
        net = slim.conv2d(net, in_channel * 4, [1, 1],
                          normalizer_fn=slim.batch_norm,
                          activation_fn=None,
                          trainable=trainable)

        if down_sample != None:
            residual = slim.conv2d(input, down_sample['out_channel'], [3, 3],
                                   stride=down_sample['stride'],
                                   normalizer_fn=slim.batch_norm,
                                   activation_fn=None,
                                   trainable=trainable)
        else:
            residual = input

        net = tf.add(residual, net)
        net = tf.nn.relu(net)
    return net


def res_block_v1(input, trainable=True):
    """
    res 版本 V1.0
                         channel    shape
        1.conv              3      256 * 3
        2.res_block         32     256 * 10
          [8,8,32]
        3.res_block         128    256 * 5
          [32,32,128]
        4.reshape           1      256 * 640

    :param input: 输入tensor
    :param trainable:
    :return:
    """
    net = slim.conv2d(input, 3, [5, 5], stride=[1, 1], trainable=trainable)

    down_sample = {
        'stride': [1, 2],
        'out_channel': 32}
    net = res_block(net, 8, stride=[1,2], down_sample=down_sample, trainable=trainable, scope='bottleneck_2_1')
    net = res_block(net, 8, trainable=trainable, scope='bottleneck_2_2')
    net = res_block(net, 8, trainable=trainable, scope='bottleneck_2_3')

    down_sample = {
        'stride': [1, 2],
        'out_channel': 128}
    net = res_block(net, 32, stride=[1,2], down_sample=down_sample, trainable=trainable, scope='bottleneck_3_1')
    net = res_block(net, 32, trainable=trainable, scope='bottleneck_3_2')
    net = res_block(net, 32, trainable=trainable, scope='bottleneck_3_3')

    net_shape = net.shape
    net = tf.reshape(net, (-1, net_shape[1], net_shape[-1]*net_shape[-2]))

    return net


if __name__ == '__main__':
    a = tf.placeholder(tf.float32, [None, 256, 20, 1])

    a = res_block_v1(a)
