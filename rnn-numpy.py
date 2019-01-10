# -*- coding: utf-8 -*-
# author: kmrocki
# based on the original code by A.Karpathy (char-rnn)

import numpy as np
import argparse, sys
import datetime, time
import random
from random import uniform

### parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', type=int, default = 16, help='batch size')
parser.add_argument('--hidden', type=int, default = 32, help='hiddens')
parser.add_argument('--seqlength', type=int, default = 25, help='seqlength')

opt = parser.parse_args()
B = opt.batchsize # batch size
S = opt.seqlength # unrolling in time steps
HN = opt.hidden # size of hidden layer of neurons
learning_rate = 1e-1
clipgrads = False

# data I/O
data = open('./alice29.txt', 'r').read()

chars = list(set(data))
data_size, M = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, M)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# controller parameters
Wxh = np.random.randn(HN, M)*0.01 # input to hidden
Whh = np.random.randn(HN, HN)*0.01 # hidden to hidden
Why = np.random.randn(M, HN)*0.01 # hidden to output
bh = np.zeros((HN, 1)) # hidden bias
by = np.zeros((M, 1)) # output bias

def train(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is HxB array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  # inputs, outputs, controller states
  xs, hs, ys, ps = {}, {}, {}, {}
  #init previous states
  hs[-1] = np.copy(hprev)

  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((M, B)) # encode in 1-of-k representation
    for b in range(0,B): xs[t][:,b][inputs[t][b]] = 1

    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars

    ###################
    mx = np.max(ys[t], axis=0)
    ys[t] -= mx # normalize
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]), axis=0) # probabilities for next chars

    for b in range(0,B):
        if ps[t][targets[t,b],b] > 0: loss += -np.log(ps[t][targets[t,b],b]) # softmax (cross-entropy loss)

  # backward pass:
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)

  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    for b in range(0,B): dy[targets[t][b], b] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += np.expand_dims(np.sum(dy,axis=1), axis=1)
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dh = dh * (1 - hs[t] * hs[t]) # backprop though tanh
    dbh += np.expand_dims(np.sum(dh,axis=1), axis=1)
    dWxh += np.dot(dh, xs[t].T)
    dWhh += np.dot(dh, hs[t-1].T)
    dhnext = np.dot(Whh.T, dh)

    if clipgrads:
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

n = 0
p = np.random.randint(len(data)-1-S,size=(B)).tolist()
inputs = np.zeros((S,B), dtype=int)
targets = np.zeros((S,B), dtype=int)
hprev = np.zeros((HN,B))
mWxh, mWhh, mWhy  = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/M)*S # loss at iteration 0
start = time.time()

t = time.time()-start
last=start
T = 1000 # max time
while t < T:
  # prepare inputs (we're sweeping from left to right in steps S long)
  for b in range(0,B):
      if p[b]+S+1 >= len(data) or n == 0:
        hprev[:,b] = np.zeros(HN) # reset hidden memory
        p[b] = np.random.randint(len(data)-1-S)

      inputs[:,b] = [char_to_ix[ch] for ch in data[p[b]:p[b]+S]]
      targets[:,b] = [char_to_ix[ch] for ch in data[p[b]+1:p[b]+S+1]]

  # forward S characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = train(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + np.mean(loss)/(np.log(2)*B) * 0.001
  if n % 10 == 0:
      tdelta = time.time()-last
      last = time.time()
      t = time.time()-start
      print '%.3f s, iter %d, %.4f BPC, %.2f char/s' % (t, n, smooth_loss / S, (B*S*10)/tdelta) # print progress

  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
  # perform parameter update with Adagrad
   mem += dparam * dparam
   param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  for b in range(0,B): p[b] += S # move data pointer
  n += 1 # iteration counter
