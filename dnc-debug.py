# -*- coding: utf-8 -*-
# author: kmrocki

from __future__ import print_function
import numpy as np
import argparse, sys
import datetime, time
import random
from random import uniform
import matplotlib.pyplot as plt
#from multiprocessing import Process, Value, Lock
import time

try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def fun_key_simil(C, K): return np.dot(C, K)
def dfun_key_simil(C, dsim): return np.dot(C.T, dsim)
def length(V): return np.sqrt(np.sum(V*V))
def normalize(V): return V/length(V)
### parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fname', type=str, default = './' + sys.argv[0] + '.log', help='log filename')
parser.add_argument('--batchsize', type=int, default = 16, help='batch size')
parser.add_argument('--hidden', type=int, default = 32, help='hiddens')
parser.add_argument('--seqlength', type=int, default = 25, help='seqlength')
parser.add_argument('--timelimit', type=int, default = 600, help='time limit (s)')
parser.add_argument('--gradcheck', action='store_const', const=True, default=False, help='run gradcheck?')
parser.add_argument('--fp64', action='store_const', const=True, default=False, help='double precision?')
parser.add_argument('--sample_length', type=int, default=500, help='sample length')
parser.add_argument('--check_interval', type=int, default=200, help='check interval (sample, grads)')

opt = parser.parse_args()
print((datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sys.argv[0], opt))
logname = opt.fname
gradchecklogname = 'gradcheck.log'
samplelogname = 'sample.log'
B = opt.batchsize
S = opt.seqlength
T = opt.timelimit
GC = opt.gradcheck
plotting = False

datatype = np.float32
if opt.fp64: datatype = np.float64



# gradient checking
def gradCheck(inputs, target, cprev, hprev, mprev, rprev):
  global Wxh, Whh, Why, Whr, Whv, Whw, Whe, Wrh, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby, _, _ = lossFun(inputs, targets, cprev, hprev, mprev, rprev)
  print('GRAD CHECK\n')
  with open(gradchecklogname, "w") as myfile: myfile.write("-----\n")

  for param,dparam,name in zip([Wxh, Whh, Why, Whr, Whv, Whw, Whe, Wrh, Wry, bh, by], [dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby], ['Wxh', 'Whh', 'Why', 'Whr', 'Whv', 'Whw', 'Whe', 'Wrh', 'Wry', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (repr(s0), repr(s1))
    min_error, mean_error, max_error = 1,0,0
    min_numerical, max_numerical = 1e10, -1e10
    min_analytic, max_analytic = 1e10, -1e10
    valid_checks = 0
    for i in xrange(num_checks):
      ri = int(uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ ,_, _, _, _ , _ , _ , _ = lossFun(inputs, targets, cprev, hprev, mprev, rprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ ,_, _, _, _ , _ , _ , _ = lossFun(inputs, targets, cprev, hprev, mprev, rprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = 0
      vdiff = abs(grad_analytic - grad_numerical)
      vsum = abs(grad_numerical + grad_analytic)
      min_numerical = min(grad_numerical, min_numerical)
      max_numerical = max(grad_numerical, max_numerical)
      min_analytic = min(grad_analytic, min_analytic)
      max_analytic = max(grad_analytic, max_analytic)

      if vsum > 0:
          rel_error = vdiff / vsum
          min_error = min(min_error, rel_error)
          max_error = max(max_error, rel_error)
          mean_error = mean_error + rel_error
          valid_checks += 1

    mean_error /= num_checks
    print('%s:\t\tn = [%e, %e]\tmin %e, max %e\t\n\t\ta = [%e, %e]\tmean %e # %d/%d' % (name, min_numerical, max_numerical, min_error, max_error, min_analytic, max_analytic, mean_error, num_checks, valid_checks))
      # rel_error should be on order of 1e-7 or less
    entry = '%s:\t\tn = [%e, %e]\tmin %e, max %e\t\n\t\ta = [%e, %e]\tmean %e # %d/%d\n' % (name, min_numerical, max_numerical, min_error, max_error, min_analytic, max_analytic, mean_error, num_checks, valid_checks)
    with open(gradchecklogname, "a") as myfile: myfile.write(entry)


start = time.time()
with open(logname, "a") as myfile:
    entry = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sys.argv[0], opt
    myfile.write("# " + str(entry))
    myfile.write("\n#  ITER\t\tTIME\t\tTRAIN LOSS\n")

# data I/O
#data = open('./ptb/ptb.train.txt', 'r').read() # should be simple plain text file
data = open('./alice29.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

clipGradients = False

# hyperparameters
HN = opt.hidden # size of hidden layer of neurons
S = opt.seqlength # number of steps to unroll the RNN for
learning_rate = 1e-1 #5*1e-2
B = opt.batchsize

# model parameters
Wxh = np.random.randn(4*HN, vocab_size).astype(datatype)*0.01 # input to hidden
Whh = np.random.randn(4*HN, HN).astype(datatype)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, HN).astype(datatype)*0.01 # hidden to output
bh = np.zeros((4*HN, 1), dtype = datatype) # hidden bias
by = np.zeros((vocab_size, 1), dtype = datatype) # output bias

# external memory
# TODO check the size constraints relative to HN
MW = 8 # paper - W
MN = 8 # paper - N
N = HN
M = vocab_size
MR = 1 # paper - R

Wrh = np.random.randn(4*HN, MW).astype(datatype)*0.01 # read vector to hidden
Whv = np.random.randn(MW, HN).astype(datatype)*0.01 # write content
Whr = np.random.randn(MW, HN).astype(datatype)*0.01 # read strength
Whw = np.random.randn(MW, HN).astype(datatype)*0.01 # write strength
Whe = np.random.randn(MW, HN).astype(datatype)*0.01 # erase strength
Wry = np.random.randn(vocab_size, MR * MW).astype(datatype)*0.01 # erase strength

# i o f c
# init f gates biases higher
bh[2*N:3*N,:] = 1

def lossFun(inputs, targets, cprev, hprev, mprev, rprev, plot=False):
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

  for t in xrange(len(inputs)):
      mem_write_gate[t] = np.zeros((MN,B), dtype=datatype)
      mem_read_gate[t] = np.zeros((MN,B), dtype=datatype)

  dmem_write_key = np.zeros((MW,B), dtype=datatype)
  dmem_read_key = np.zeros((MW,B), dtype=datatype)

  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size, B), dtype=datatype) # encode in 1-of-k representation
    for b in range(0,B): xs[t][:,b][inputs[t][b]] = 1

    gs[t] = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh # gates, linear part

    ############ external memory input ##########
    gs[t] += np.dot(Wrh, rs[t-1]) # add previous read vector

    # gates nonlinear part
    gs[t][0:3*N,:] = sigmoid(gs[t][0:3*N,:]) #i, o, f gates
    gs[t][3*N:4*N, :] = np.tanh(gs[t][3*N:4*N,:]) #c gate

    #mem(t) = c gate * i gate + f gate * mem(t-1)
    cs[t] = gs[t][3*N:4*N,:] * gs[t][0:N,:] + gs[t][2*N:3*N,:] * cs[t-1]
    cs[t] = np.tanh(cs[t]) # mem cell - nonlinearity
    hs[t] = gs[t][N:2*N,:] * cs[t] # new hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars

    ##### external mem ########
    mem_read_key[t] = np.dot(Whr, hs[t]) # key used for content based read
    mem_write_key[t] = np.dot(Whw, hs[t]) # key used for content based read

    mem_new_content[t] = np.dot(Whv, hs[t])

    for b in range(0,B):
        # normalize - unit length
        #mem_read_key[t][:,b] = np.exp(mem_read_key[t][:,b]) / np.sum(np.exp(mem_read_key[t][:,b]), axis=0) # probabilities for next chars
        #mem_write_key[t][:,b] = np.exp(mem_write_key[t][:,b]) / np.sum(np.exp(mem_write_key[t][:,b]), axis=0) # probabilities for next chars

        #s = np.sum(np.exp(memory[t-1][:,:,b]), axis=1, keepdims=1)
        #memory[t-1][:,:,b] = np.exp(memory[t-1][:,:,b])/s

        mem_write_gate[t][:,b,None] = np.dot(memory[t-1][:,:,b], mem_write_key[t][:,b,None])
        mem_read_gate[t][:,b,None] = np.dot(memory[t-1][:,:,b], mem_read_key[t][:,b,None])

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
    ys[t] -= mx
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]), axis=0) # probabilities for next chars

    for b in range(0,B):
        if ps[t][targets[t,b],b] > 0: loss += -np.log(ps[t][targets[t,b],b]) # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why )

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

    ##external######
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

    for b in range(0,B):
        dmem_next[:,:,b] += np.dot(dmem_read_gate[:,:,b].T, mem_read_key[t][:,b,None].T)
        dmem_next[:,:,b] += np.dot(dmem_write_gate[:,:,b].T, mem_write_key[t][:,b,None].T)

        #  dmem_next[:,:,b] = dmem_next[:,:,b] * memory[t-1][:,:,b]
        #  dmem_next_sum = np.sum(dmem_next[:,:,b], axis=1, keepdims=1)
        #  dmem_next[:,:,b] -= memory[t-1][:,:,b] * dmem_next_sum

    #  dmem_read_key = dmem_read_gate
    dmem_erase_gate = dmem_erase_gate * mem_erase_gate[t] * (1-mem_erase_gate[t])

    dmem_write_gate = np.reshape(dmem_write_gate, (MN,B))
    dmem_read_gate = np.reshape(dmem_read_gate, (MN,B))

    for b in range(0,B):
        dmem_write_key[:,b,None] = np.dot(memory[t-1][:,:,b].T, dmem_write_gate[:,b,None])
        dmem_read_key[:,b,None] = np.dot(memory[t-1][:,:,b].T, dmem_read_gate[:,b,None])

        #  dmem_read_key[:,b,None] = dmem_read_key[:,b,None] * mem_read_key[t][:,b,None]
        #  dmem_read_key_sum = np.sum(dmem_read_key[:,b,None], axis=0)
        #  dmem_read_key[:,b,None] -= mem_read_key[t][:,b,None] * dmem_read_key_sum

        #  dmem_write_key[:,b,None] = dmem_write_key[:,b,None] * mem_write_key[t][:,b,None]
        #  dmem_write_key_sum = np.sum(dmem_write_key[:,b,None], axis=0)
        #  dmem_write_key[:,b,None] -= mem_write_key[t][:,b,None] * dmem_write_key_sum


    dWhw += np.dot(np.reshape(dmem_write_key, (MW, B)), hs[t].T)
    dWhr += np.dot(np.reshape(dmem_read_key, (MW,B)), hs[t].T)
    dWhe += np.dot(np.sum(dmem_erase_gate, axis=0), hs[t].T)
    dWhv += np.dot(np.sum(dmem_new_content, axis=0), hs[t].T)
    ########

    dby += np.expand_dims(np.sum(dy,axis=1), axis=1)
    dh = np.dot(Why.T, dy) + dhnext # backprop into h

    dh += np.dot(Whw.T, np.reshape(dmem_write_key, (MW,B)))
    dh += np.dot(Whr.T, np.reshape(dmem_read_key, (MW,B)))
    dh += np.dot(Whe.T, np.sum(dmem_erase_gate, axis=0))
    dh += np.dot(Whv.T, np.sum(dmem_new_content, axis=0))
    # external end ###

    dc = dh * gs[t][N:2*N,:] + dcnext # backprop into c
    dc = dc * (1 - cs[t] * cs[t]) # backprop though tanh

    dg[N:2*N,:] = dh * cs[t] # o gates
    dg[0:N,:] = gs[t][3*N:4*N,:] * dc # i gates
    dg[2*N:3*N,:] = cs[t-1] * dc # f gates
    dg[3*N:4*N,:] = gs[t][0:N,:] * dc # c gates
    dg[0:3*N,:] = dg[0:3*N,:] * gs[t][0:3*N,:] * (1 - gs[t][0:3*N,:]) # backprop through sigmoids
    dg[3*N:4*N,:] = dg[3*N:4*N,:] * (1 - gs[t][3*N:4*N,:] * gs[t][3*N:4*N,:]) # backprop through tanh
    dbh += np.expand_dims(np.sum(dg,axis=1), axis=1)
    dWxh += np.dot(dg, xs[t].T)
    dWhh += np.dot(dg, hs[t-1].T)
    dWrh += np.dot(dg, rs[t-1].T)
    dhnext = np.dot(Whh.T, dg)
    drs_next = np.dot(Wrh.T, dg)
    dcnext = dc * gs[t][2*N:3*N,:]
    
  if plotting and plot:
      _b_ = 1 # sequence number
      plt.subplot(2,1,1)
      plt.imshow(memory[MW][:,:,_b_])
      plt.colorbar()
      plt.title('memory state')
      plt.subplot(2,1,2)
      read_gates_history = np.zeros((S, MN))
      for i in range(0,S): read_gates_history[i,:] = mem_read_gate[i][:,_b_]
      plt.imshow(read_gates_history.T)
      cbar = plt.colorbar()
      cbar.ax.get_yaxis().labelpad = 20
      cbar.ax.set_ylabel('activation', rotation=270)
      plt.ylabel('location')
      plt.yticks(np.arange(0, MN))
      plt.xlabel('time step')
      plt.title('mem read gate in time')
      plt.show()
      plot = False

  if clipGradients:
    for dparam in [dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby]:
      np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

  return loss, dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby, cs[len(inputs)-1], hs[len(inputs)-1]

def sample(c, h, m, r, seed_ix, n):
  """
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1), dtype=datatype)
  x[seed_ix] = 1
  ixes = []

  for t in xrange(n):
    g = np.dot(Wxh, x) + np.dot(Whh,h) + np.dot(Wrh, r) + bh
    g[0:3*N,:] = sigmoid(g[0:3*N,:])
    g[3*N:4*N, :] = np.tanh(g[3*N:4*N,:])
    c = g[3*N:4*N,:] * g[0:N,:] + g[2*N:3*N,:] * c
    c = np.tanh(c)
    h = g[N:2*N,:] * c
    y = np.dot(Why, h) + by
    mem_new_content = np.dot(Whv, h)
    #mem_write_gate = sigmoid(np.dot(Whw, h))
    #mem_read_gate = sigmoid(np.dot(Whr, h))
    mem_write_key = np.dot(Whw, h)
    mem_read_key = np.dot(Whr, h)
    mem_write_gate = np.exp(mem_write_key)
    mem_read_gate = np.exp(mem_read_key)
    mem_write_gate_sum = np.sum(mem_write_gate, axis=0)
    mem_read_gate_sum = np.sum(mem_read_gate, axis=0)
    mem_write_gate = mem_write_gate/mem_write_gate_sum
    mem_read_gate = mem_read_gate/mem_read_gate_sum
    mem_erase_gate = sigmoid(np.dot(Whe, h))
    #print m.shape
    m = m * (1-np.reshape(mem_erase_gate,(1,MW)) * np.reshape(mem_write_gate, (MN,1))) # 1
    #  m = m * (1-mem_erase_gate) # 2
    m += np.reshape(mem_new_content,(1,MW)) * np.reshape(mem_write_gate, (MN,1))
    r = np.expand_dims(np.sum(m * np.reshape(mem_read_gate, (MN, 1)), axis=0), axis=1)
    y += np.dot(Wry, r)
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

if __name__ == "__main__":

    v = 0
    n = 0
    p = np.random.randint(len(data)-1-S,size=(B)).tolist()
    inputs = np.zeros((S,B), dtype=int)
    targets = np.zeros((S,B), dtype=int)
    cprev = np.zeros((HN,B), dtype=datatype)
    hprev = np.zeros((HN,B), dtype=datatype)
    mprev = np.zeros((MN,MW,B), dtype=datatype)
    rprev = np.zeros((MW,B), dtype=datatype)
    mWxh, mWhh, mWhy  = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mWhr, mWhv, mWhw, mWhe, mWrh, mWry = np.zeros_like(Whr), np.zeros_like(Whv), np.zeros_like(Whw), np.zeros_like(Whe), np.zeros_like(Wrh), np.zeros_like(Wry)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/vocab_size)*S # loss at iteration 0
    start = time.time()


    t = 0
    last=start
    while t < T:
      # prepare inputs (we're sweeping from left to right in steps S long)
      for b in range(0,B):
          if p[b]+S+1 >= len(data) or n == 0:
            cprev[:,b] = np.zeros(HN, dtype=datatype) # reset LSTM memory
            hprev[:,b] = np.zeros(HN, dtype=datatype) # reset hidden memory
            mprev[:,:,b] = np.random.randn(MN,MW)*0.01 # reset ext memory
            rprev[:,b] = np.zeros(MW, dtype=datatype) # reset read vec memory
            p[b] = np.random.randint(len(data)-1-S)

          Pb = p[b]
          inputs[:,b] = [char_to_ix[ch] for ch in data[Pb:Pb+S]]
          targets[:,b] = [char_to_ix[ch] for ch in data[Pb+1:Pb+S+1]]

      # sample from the model now and then
      if (n+1) % opt.check_interval == 0:
          #if log..
          sample_ix = sample(np.expand_dims(cprev[:,0], axis=1), np.expand_dims(hprev[:,0], axis=1), (mprev[:,:,0]), np.expand_dims(rprev[:,0], axis=1), inputs[0], opt.sample_length)
          txt = ''.join(ix_to_char[ix] for ix in sample_ix)
          print('----\n %s \n----' % (txt, ))
          entry = '%s\n' % (txt)
          with open(samplelogname, "w") as myfile: myfile.write(entry)
          
          if (GC):
              gradCheck(inputs, targets, cprev, hprev, mprev, rprev)

      plot = n % opt.check_interval == 200 and n > 0

      # forward S characters through the net and fetch gradient
      loss, dWxh, dWhh, dWhy, dWhr, dWhv, dWhw, dWhe, dWrh, dWry, dbh, dby, cprev, hprev = lossFun(inputs, targets, cprev, hprev, mprev, rprev, plot)
      smooth_loss = smooth_loss * 0.999 + np.mean(loss)/(np.log(2)*B) * 0.001
      interval = time.time() - last

      if n % opt.check_interval == 0 and n > 0:
        #if log..
        tdelta = time.time()-last
        last = time.time()
        t = time.time()-start
        entry = '{:5}\t\t{:3f}\t{:3f}\n'.format(n, t, smooth_loss/S)
        with open(logname, "a") as myfile: myfile.write(entry)

        print('%2d: %.3f s, iter %d, %.4f BPC, %.2f char/s' % (v, t, n, smooth_loss / S, (B*S*100)/tdelta)) # print progress

      COMPONENTS = [ ( Wxh, dWxh, mWxh), (Whh,dWhh,mWhh), (Why,dWhy,mWhy), (Whr,dWhr,mWhr), (Whv,dWhv,mWhv), (Whe,dWhe,mWhe), (Wrh,dWrh,mWrh), (Wry,dWry,mWry), (bh,dbh,mbh), (by,dby,mby)  ]
      for param, dparam, mem in COMPONENTS:
        # perform parameter update with Adagrad
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      for b in range(0,B): p[b] += S # move data pointer
      n += 1 # iteration counter

#procs = [Process(target=proc, args=(i,)) for i in range(0,8)]
#for p in procs: 
#    p.start()
#    time.sleep(0.5)
#for p in procs: p.join()
