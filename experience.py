import cPickle
import numpy as np
import os
import random

class Experience(object):
  def __init__(self, config):
    self.config = config
    self.memory = []
    self.reward_history = []

  def reset(self):
    self.memory = []

  def add(self, state, action, reward):
    self.memory.append([state, action, reward])

  def getState(self, index):
    raise NotImplementedError

  def getLastestState(self):
    return self.memory[-1][0]

  def sample(self):
    future_reward = 0
    r = 0
    for i in reversed(xrange(len(self.memory))):
      r = self.memory[i][-1] + self.config.GAMMA * r
      self.memory[i][-1] = r
      self.reward_history.append(r)

    self.reward_history = self.reward_history[max(0, len(self.reward_history) -
        self.config.REWARD_HISTORY_SIZE + 1):]
    reward_mean = np.mean(self.reward_history)
    reward_stddev = np.std(self.reward_history)

    for t in xrange(len(self.memory)):
      self.memory[t][-1] = (self.memory[t][-1] - reward_mean) / reward_stddev
      yield self.memory[t]



