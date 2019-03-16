#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: demo.py

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

data = np.arange(21)
anchor, positive, negative = tf.unstack(tf.reshape(data, [-1, 3, 1]), 3, 1)

print(anchor)
print(positive)
print(negative)

