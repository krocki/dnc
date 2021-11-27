#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# author: kmrocki
# some parts still missing:
# - multiple heads
# - dynamic allocation/free
# - key strengths (betas)
# - some parts need to be profiled/optimized
# - put the code into some kind of classes, connect with other SW

from __future__ import print_function
import numpy as np
import argparse, sys
import datetime, time
import random
from random import uniform

try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

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

# external memory
MR = 1 # memory read heads (paper - R)
MW = 10 # memory entry width (paper - W)
MN = 5 # number of memory locations (paper - N)

# data I/O
data = open('./alice29.txt', 'r').read() # should be simple plain text file

chars = list(set(data))
data_size, M = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, M))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# controller parameters
Wxh = np.random.randn(4*HN, M)*0.01 # input to hidden
Whh = np.random.randn(4*HN, HN)*0.01 # hidden to hidden
Why = np.random.randn(M, HN)*0.01 # hidden to output
bh = np.zeros((4*HN, 1)) # hidden bias
by = np.zeros((M, 1)) # output bias

Wrh = np.random.randn(4*HN, MW)*0.01 # read vector to hidden
Whv = np.random.randn(MW, HN)*0.01 # write content
Whr = np.random.randn(MN, HN)*0.01 # read strength
Whw = np.random.randn(MN, HN)*0.01 # write strength
Whe = np.random.randn(MW, HN)*0.01 # erase strength
Wry = np.random.randn(M, MR * MW)*0.01 # erase strength

# init LSTM f gates biases higher
bh[2*HN:3*HN,:] = 1

def train(inputs, targets, cprev, hprev, mprev, rprev):
  """
  inputs,targets are both list of integers.
  cprev is HxB array of initial memory cell state
  hprev is HxB array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  # inputs, outputs, controller states
  xs, hs, ys, ps, gs, cs = {}, {}, {}, {}, {}, {}
  # external mem
  mem_new_content, mem_read_gate, mem_write_gate, mem_erase_gate, memory, rs = {}, {}, {}, {}, {}, {}
  mem_read_key, mem_write_key = {}, {}
  #init previous states
  hs[-1], cs[-1], rs[-1], memory[-1] = np.copy(hprev), np.copy(cprev), np.copy(rprev), np.copy(mprev)

  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((M, B)) # encode in 1-of-k representation
    for b in range(0,B): xs[t][:,b][inputs[t][b]] = 1

    gs[t] = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh # gates, linear part

    ############ external memory input ##########
    gs[t] += np.dot(Wrh, rs[t-1]) # add previous read vector
    ############

    ####### LSTM controller
    # gates nonlinear part
    gs[t][0:3*HN,:] = sigmoid(gs[t][0:3*HN,:]) #i, o, f gates
    gs[t][3*HN:4*HN, :] = np.tanh(gs[t][3*HN:4*HN,:]) #c gate
    #mem(t) = c gate * i gate + f gate * mem(t-1)
    cs[t] = gs[t][3*HN:4*HN,:] * gs[t][0:HN,:] + gs[t][2*HN:3*HN,:] * cs[t-1]
    cs[t] = np.tanh(cs[t]) # mem cell - nonlinearity
    hs[t] = gs[t][HN:2*HN,:] * cs[t] # new hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ###################

    ##### external mem ########
    mem_read_key[t] = np.dot(Whr, hs[t]) # key used for content based read
    mem_write_key[t] = np.dot(Whw, hs[t]) # key used for content based read

    mem_new_content[t] = np.dot(Whv, hs[t])
    mem_write_gate[t] = mem_write_key[t]
    mem_read_gate[t] = mem_read_key[t]
    mem_erase_gate[t] = sigmoid(np.dot(Whe, hs[t]))

    #softmax on read and write gates
    mem_write_gate[t] = np.exp(mem_write_gate[t])
    mem_write_gate_sum = np.sum(mem_write_gate[t], axis=0)
    mem_write_gate[t] = mem_write_gate[t]/mem_write_gate_sum

    mem_read_gate[t] = np.exp(mem_read_gate[t])
    mem_read_gate_sum = np.sum(mem_read_gate[t], axis=0)
    mem_read_gate[t] = mem_read_gate[t]/mem_read_gate_sum
    ######

    memory[t] = memory[t-1] * (1 - np.reshape(mem_erase_gate[t], (1, MW, B)) * np.reshape(mem_write_gate[t], (MN, 1, B)))
    memory[t] += np.reshape(mem_new_content[t], (1, MW, B)) * np.reshape(mem_write_gate[t], (MN, 1, B))

    rs[t] = memory[t] * np.reshape(mem_read_gate[t], (MN, 1, B))
    rs[t] = np.sum(rs[t], axis=0)

    ys[t] += np.dot(Wry, rs[t]) # add read vector to output

    ###########################
    mx = np.max(ys[t], axis=0)
    ys[t] -= mx # normalize
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]), axis=0) # probabilities for next chars

    for b in range(0,B):
        if ps[t][targets[t,b],b] > 0: loss += -np.log(ps[t][targets[t,b],b]) # softmax (cross-entropy loss)

  # backward pass:
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  ### ext
  dWhr, dWhv, dWhw, dWhe, dWrh, dWry = np.zeros_like(Whr), np.zeros_like(Whv), np.zeros_like(Whw), np.zeros_like(Whe), np.zeros_like(Wrh), np.zeros_like(Wry)
  ###

  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dcnext = np.zeros_like(cs[0])
  dhnext = np.zeros_like(hs[0])
  dmemory = np.zeros_like(memory[0])
  dmem_next = np.zeros_like(memory[0])
  drs_next = np.zeros_like(rs[0])
  dg = np.zeros_like(gs[0])
  W_ones = np.ones((MW,1))

  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    for b in range(0,B): dy[targets[t][b], b] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)

    ## external memory ######
    dWry += np.dot(dy, rs[t].T)
    drs = np.dot(Wry.T, dy) + drs_next
    dmemory = np.reshape(drs, (1, MW, B)) * np.reshape(mem_read_gate[t], (MN, 1, B)) + dmem_next

    #iface gates
    dmem_write_gate = np.dot(W_ones.T, dmemory * (mem_new_content[t] - mem_erase_gate[t] * memory[t-1])) # 1
    dmem_read_gate = np.dot(W_ones.T, drs * memory[t])

    # propagate back through softmax
    dmem_write_gate = dmem_write_gate * mem_write_gate[t]
    dmem_write_gate_sum = np.sum(dmem_write_gate, axis=1)
    dmem_write_gate -= mem_write_gate[t] * dmem_write_gate_sum
    dmem_read_gate = dmem_read_gate * mem_read_gate[t]
    dmem_read_gate_sum = np.sum(dmem_read_gate, axis=1)
    dmem_read_gate -= mem_read_gate[t] * dmem_read_gate_sum

    dmem_new_content = dmemory * np.reshape(mem_write_gate[t], (MN, 1, B))
    dmem_erase_gate = -dmemory * memory[t-1] * np.reshape(mem_write_gate[t], (MN, 1, B))
    dmem_next = dmemory * (1 - np.reshape(mem_erase_gate[t], (1, MW, B)) * np.reshape(mem_write_gate[t], (MN,1,B)))

    dmem_read_key = dmem_read_gate
    dmem_write_key = dmem_write_gate
    dmem_erase_gate = dmem_erase_gate * mem_erase_gate[t] * (1-mem_erase_gate[t])

    # go back to linearities
    dWhw += np.dot(np.reshape(dmem_write_key, (MN, B)), hs[t].T)
    dWhr += np.dot(np.reshape(dmem_read_key, (MN,B)), hs[t].T)
    dWhe += np.dot(np.sum(dmem_erase_gate, axis=0), hs[t].T)
    dWhv += np.dot(np.sum(dmem_new_content, axis=0), hs[t].T)

    dby += np.expand_dims(np.sum(dy,axis=1), axis=1)
    dh = np.dot(Why.T, dy) + dhnext # backprop into h

    dh += np.dot(Whw.T, np.reshape(dmem_write_key, (MN,B)))
    dh += np.dot(Whr.T, np.reshape(dmem_read_key, (MN,B)))
    dh += np.dot(Whe.T, np.sum(dmem_erase_gate, axis=0))
    dh += np.dot(Whv.T, np.sum(dmem_new_content, axis=0))
    ### external memory backprop end ###

    # LSTM part
    dc = dh * gs[t][HN:2*HN,:] + dcnext # backprop into c
    dc = dc * (1 - cs[t] * cs[t]) # backprop though tanh

    dg[HN:2*HN,:] = dh * cs[t] # o gates
    dg[0:HN,:] = gs[t][3*HN:4*HN,:] * dc # i gates
    dg[2*HN:3*HN,:] = cs[t-1] * dc # f gates
    dg[3*HN:4*HN,:] = gs[t][0:HN,:] * dc # c gates
    dg[0:3*HN,:] = dg[0:3*HN,:] * gs[t][0:3*HN,:] * (1 - gs[t][0:3*HN,:]) # backprop through sigmoids
    dg[3*HN:4*HN,:] = dg[3*HN:4*HN,:] * (1 - gs[t][3*HN:4*HN,:] * gs[t][3*HN:4*HN,:]) # backprop through tanh
    dbh += np.expand_dims(np.sum(dg,axis=1), axis=1)
    dWxh += np.dot(dg, xs[t].T)
    dWhh += np.dot(dg, hs[t-1].T)
    dWrh += np.dot(dg, rs[t-1].T)
    dhnext = np.dot(Whh.T, dg)
    drs_next = np.dot(Wrh.T, dg)
    dcnext = dc * gs[t][2*HN:3*HN,:]

    if clipgrads:
        for dparam in [dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby, cs[len(inputs)-1], hs[len(inputs)-1]

n = 0
p = np.random.randint(len(data)-1-S,size=(B)).tolist()
inputs = np.zeros((S,B), dtype=int)
targets = np.zeros((S,B), dtype=int)
cprev = np.zeros((HN,B))
hprev = np.zeros((HN,B))
mprev = np.zeros((MN,MW,B))
rprev = np.zeros((MW,B))
mWxh, mWhh, mWhy  = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mWhr, mWhv, mWhw, mWhe, mWrh, mWry = np.zeros_like(Whr), np.zeros_like(Whv), np.zeros_like(Whw), np.zeros_like(Whe), np.zeros_like(Wrh), np.zeros_like(Wry)
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
        cprev[:,b] = np.zeros(HN) # reset LSTM memory
        hprev[:,b] = np.zeros(HN) # reset hidden memory
        mprev[:,:,b] = np.zeros((MN,MW)) # reset ext memory
        rprev[:,b] = np.zeros(MW) # reset read vec memory
        p[b] = np.random.randint(len(data)-1-S)

      inputs[:,b] = [char_to_ix[ch] for ch in data[p[b]:p[b]+S]]
      targets[:,b] = [char_to_ix[ch] for ch in data[p[b]+1:p[b]+S+1]]

  # forward S characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby, cprev, hprev = train(inputs, targets, cprev, hprev, mprev, rprev)
  smooth_loss = smooth_loss * 0.999 + np.mean(loss)/(np.log(2)*B) * 0.001
  if n % 10 == 0:
      tdelta = time.time()-last
      last = time.time()
      t = time.time()-start
      print('%.3f s, iter %d, %.4f BPC, %.2f char/s' % (t, n, smooth_loss / S, (B*S*10)/tdelta)) # print progress
  
  for param, dparam, mem in zip([Wxh, Whh, Why, Whr, Whv, Whw, Whe, Wrh, Wry, bh, by],
                                [dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby], 
                                [mWxh, mWhh, mWhy, mWhr, mWhv, mWhw, mWhe, mWrh, mWry, mbh, mby]):
  # perform parameter update with Adagrad
   mem += dparam * dparam
   param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  for b in range(0,B): p[b] += S # move data pointer
  n += 1 # iteration counter
