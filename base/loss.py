#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: loss.py

import tensorflow as tf

def triplet_loss(anchor, positive, negative, alpha):
    """
    facenet triplet loss (reference to facenet)
    :param anchor:
    :param positive:
    :param negative:
    :param alpha:
    :return:
    """
    with tf.variable_scope('triplet_loss'):
        # positive distance
        pos_dist = tf.reduce_mean(
                    tf.square(
                    tf.subtract(anchor, positive)), 1)
        # negative distance
        neg_dist = tf.reduce_mean(
                    tf.square(
                    tf.subtract(anchor, negative)), 1)

        # loss
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


def softmax_loss(logits, labels):
    """
    softmax loss
    :param logits:
    :param labels:
    :return:
    """
    with tf.variable_scope('softmax_loss'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=labels))
    return loss

