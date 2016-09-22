import tensorflow as tf
import tensorflow.contrib.slim as slim

import collections

class MLPModel(object):
  def __init__(self, config):
    self.config = config
    super(MLPModel, self).__init__()

  def arg_scope(self, reuse=None, is_training=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(1e-4),
                        activation_fn=tf.nn.relu,
                        reuse=reuse) as sc:
        return sc

  def compute(self, inputs):
    inputs = tf.reshape(inputs, shape=[-1, self.config.INPUT_DIM])
    outputs = slim.fully_connected(inputs, 20, scope="fc_0")
    outputs = slim.fully_connected(inputs, 20, scope="fc_1")
    outputs = slim.fully_connected(inputs, 20, scope="fc_2")
    outputs = slim.fully_connected(
        outputs, self.config.NUM_ACTIONS, activation_fn=None, scope="softmax")
    return outputs

