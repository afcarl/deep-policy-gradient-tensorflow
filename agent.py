import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

import numpy as np
import random

class Agent(object):
  def __init__(self, model=None):
    self.model = model
    self.config = self.model.config
    self.train_op = self.createTrainingOp()
    self.action_probs_op = self.createActionProbsOp()

  def createActionProbsOp(self):
    self.s_for_action_probs = tf.placeholder(tf.float32,
        shape=[1, self.config.INPUT_DIM])
    with slim.arg_scope(self.model.arg_scope(reuse=True)):
      logits = self.model.compute(self.s_for_action_probs)
    return tf.nn.softmax(logits)

  def createTrainingOp(self):
    self.s = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE,
                                               self.config.INPUT_DIM])
    self.a = tf.placeholder(tf.int32, shape=[self.config.BATCH_SIZE])
    self.r = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE])
    with slim.arg_scope(self.model.arg_scope(reuse=False)):
      logits = self.model.compute(self.s)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.a)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(self.config.ALPHA0)
    gradients = optimizer.compute_gradients(loss)

    for i, (grad, var) in enumerate(gradients):
      if grad is not None:
        gradients[i] = (grad * self.r, var)

    global_step=slim.get_or_create_global_step()
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)

    return train_op


  def act(self, sess, state, random_action, epsilon):
    state = state.reshape(1, self.config.INPUT_DIM)
    action_probs = sess.run(self.action_probs_op,
        feed_dict={ self.s_for_action_probs : state })[0]

    if random.random() < epsilon:
      return random_action
    return np.random.choice(action_probs.shape[0], 1, p=action_probs)[0]

  def observe(self, sess, state, action, reward):
    sess.run(self.train_op, feed_dict={ self.s : [state], self.a : [action], self.r : [reward] })

