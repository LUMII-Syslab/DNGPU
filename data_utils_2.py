# This code is adapted from https://github.com/tensorflow/models/tree/master/neural_gpu
# Original work Copyright 2015 Google Inc. All Rights Reserved.
# Modified work Copyright (c) 2016 Institute of Mathematics and Computer Science, Latvia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Improving the Neural GPU Architecture for Algorithm Learning"""

import math
import random
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

bins = []

all_tasks = ["sort", "kvsort", "id", "rev", "rev2", "incr", "add", "left",
             "right", "bmul", "mul", "dup",
             "badd", "qadd", "search", "qmul", "mulbcd"]
forward_max = 401
log_filename = ""


train_counters = np.zeros(10000, dtype=np.int32)
test_counters = np.zeros(10000, dtype=np.int32)

def resetCounters():
  global train_counters
  global test_counters
  train_counters = np.zeros(10000, dtype=np.int32)
  test_counters = np.zeros(10000, dtype=np.int32)

resetCounters()

def pad(l):
  for b in bins:
    if b >= l: return b
  return forward_max


train_set = {}
test_set = {}
for some_task in all_tasks:
  train_set[some_task] = []
  test_set[some_task] = []
  for all_max_len in range(10000):
    train_set[some_task].append([])
    test_set[some_task].append([])

def collectBins():
  max_length = bins[-1]
  for some_task in all_tasks:
    for l in range(max_length):
      bin_length =pad(l)
      if bin_length != l:
        cur_train = train_set[some_task]
        cur_test = test_set[some_task]
        #cur_train[bin_length]+=[pad_to(inp, bin_length)  for inp in cur_train[l]]
        #cur_test[bin_length]+=[pad_to(inp, bin_length) for inp in cur_test[l]]
        cur_train[bin_length]+=cur_train[l]
        cur_test[bin_length] += cur_test[l]
        cur_train[l] = []
        cur_test[l] = []

  #add some shorter instances to train for padding
  for some_task in all_tasks:
    for ind in range(1,len(bins)):
      small_count = len(train_set[some_task][bins[ind]])//20 # 5% shorter instances
      for itemNr in range(small_count):
        smaller_bin = bins[random.randint(0,ind-1)]
        assert len(train_set[some_task][smaller_bin])>0
        item = random.choice(train_set[some_task][smaller_bin])
        train_set[some_task][bins[ind]].append(item)

  #shuffle randomly
  for some_task in all_tasks:
    for l in bins:
      random.shuffle(train_set[some_task][l])
      random.shuffle(test_set[some_task][l])

def to_base(num, b, l=1):
  assert num >= 0
  ans = []
  while num:
    ans.append(num%b)
    num //= b
  while len(ans) < l:
    ans.append(0)
  return ans

def tobcd(num):
  res = []
  for digit in num:
    bin_digit = to_base(digit,2,4)
    bin_digit[3]+=2 # digit end marker
    res+=bin_digit
  return res

def add(n1, n2, base=10):
  """Add two numbers represented as lower-endian digit lists."""
  k = max(len(n1), len(n2)) + 1
  d1 = n1 + [0 for _ in range(k - len(n1))]
  d2 = n2 + [0 for _ in range(k - len(n2))]
  res = []
  carry = 0
  for i in range(k):
    if d1[i] + d2[i] + carry < base:
      res.append(d1[i] + d2[i] + carry)
      carry = 0
    else:
      res.append(d1[i] + d2[i] + carry - base)
      carry = 1
  while res and res[-1] == 0:
    res = res[:-1]
  if res: return res
  return [0]

def init_data(task, length, nbr_cases, nclass):
  init_data_1(task, length, nbr_cases, nclass, train_set)
  init_data_1(task, length, nbr_cases, nclass, test_set)

"""Data initialization."""
def rand_pair(l, task):
  """Random data pair for a task. Total length should be <= l."""
  k = (l-1)//2
  if task == "mulbcd": k=(l-1)//8
  base = 10
  if task[0] == "b": base = 2
  if task[0] == "q": base = 4
  d1 = [np.random.randint(base) for _ in range(k)]
  d2 = [np.random.randint(base) for _ in range(k)]
  if task in ["add", "badd", "qadd"]:
    res = add(d1, d2, base)
  elif task in ["mul", "bmul", "qmul", "mulbcd"]:
    d1n = sum([d * (base ** i) for i, d in enumerate(d1)])
    d2n = sum([d * (base ** i) for i, d in enumerate(d2)])
    if task == "bmul":
      #res = [int(x) for x in list(reversed(str(bin(d1n * d2n))))[:-2]]
      res = to_base(d1n * d2n, base, l)
    elif task == "mul":
      res = [int(x) for x in list(reversed(str(d1n * d2n)))]
    elif task == "qmul":
      res = to_base(d1n * d2n, base,l)
    elif task == "mulbcd":
      res = to_base(d1n * d2n, base,k*2)
      res = tobcd(res)
      d1 = tobcd(d1)
      d2 = tobcd(d2)
  else:
    sys.exit()
  sep = [12]
  if task in ["add", "badd", "qadd"]: sep = [11]
  inp = [d + 1 for d in d1] + sep + [d + 1 for d in d2]
  return inp, [r + 1 for r in res]

def rand_dup_pair(l, nclass):
  """Random data pair for duplication task. Total length should be <= l."""
  k = l//2
  x = [np.random.randint(nclass - 1) + 1 for _ in range(k)]
  inp = x + [0 for _ in range(l - k)]
  res = x + x + [0 for _ in range(l - 2*k)]
  return inp, res

def rand_rev2_pair(l, nclass):
  """Random data pair for reverse2 task. Total length should be <= l."""
  inp = [(np.random.randint(nclass - 1) + 1,
          np.random.randint(nclass - 1) + 1) for _ in range(l//2)]
  res = [i for i in reversed(inp)]
  return [x for p in inp for x in p], [x for p in res for x in p]

def rand_search_pair(l, nclass):
  """Random data pair for search task. Total length should be <= l."""
  inp = [(np.random.randint(nclass - 1) + 1,
          np.random.randint(nclass - 1) + 1) for _ in range(l-1//2)]
  q = np.random.randint(nclass - 1) + 1
  res = 0
  for (k, v) in reversed(inp):
    if k == q:
      res = v
  return [x for p in inp for x in p] + [q], [res]

def rand_kvsort_pair(l, nclass):
  """Random data pair for key-value sort. Total length should be <= l."""
  keys = [(np.random.randint(nclass - 1) + 1, i) for i in range(l//2)]
  vals = [np.random.randint(nclass - 1) + 1 for _ in range(l//2)]
  kv = [(k, vals[i]) for (k, i) in keys]
  sorted_kv = [(k, vals[i]) for (k, i) in sorted(keys)]
  return [x for p in kv for x in p], [x for p in sorted_kv for x in p]

def spec(inp, task, nclass):
  """Return the target given the input for some tasks."""
  if task == "sort":
    return sorted(inp)
  elif task == "id":
    return inp
  elif task == "rev":
    return [i for i in reversed(inp)]
  elif task == "incr":
    carry = 1
    res = []
    for i in range(len(inp)):
      if inp[i] + carry < nclass:
        res.append(inp[i] + carry)
        carry = 0
      else:
        res.append(1)
        carry = 1
    return res
  elif task == "left":
    return [inp[0]]
  elif task == "right":
    return [inp[-1]]
  elif task == "left-shift":
    return [inp[l-1] for l in range(len(inp))]
  elif task == "right-shift":
    return [inp[l+1] for l in range(len(inp))]
  else:
    print_out("Unknown spec for task " + str(task))
    sys.exit()

def get_input_output_pair(l,task,nclass):
    if task in ["add", "badd", "qadd", "bmul", "mul", "qmul","mulbcd"]:
      i, t = rand_pair(l, task)
    elif task == "dup":
      i, t = rand_dup_pair(l,nclass)
    elif task == "rev2":
      i, t = rand_rev2_pair(l,nclass)
    elif task == "search":
      i, t = rand_search_pair(l,nclass)
    elif task == "kvsort":
      i, t = rand_kvsort_pair(l,nclass)
    else:
      i = [np.random.randint(nclass - 1) + 1 for ii in range(l)]
      t = spec(i,task, nclass)
    return i,t

def init_data_1(task, length, nbr_cases, nclass, cur_set):
  cur_set[task][length] = []
  l = length
  cur_time = time.time()
  total_time = 0.0
  inputSet = set()
  case_count = 0
  trials = 0

  while case_count < nbr_cases and trials < 20:
    total_time += time.time() - cur_time
    cur_time = time.time()
    if l > 10000 and case_count % 100 == 1:
      print_out("  avg gen time %.4f s" % (total_time / float(case_count)))

    i,t = get_input_output_pair(l,task,nclass)

    i_tuple = tuple(i)
    if len(i)==l and not i_tuple in inputSet:
    #if not i_tuple in inputSet:
      inputSet.add(i_tuple)
      cur_set[task][len(i)].append([i, t])
      case_count += 1
      trials = 0
    else:
      trials += 1


def to_symbol(i):
  """Covert ids to text."""
  if i == 0: return ""
  if i == 11: return "+"
  if i == 12: return "*"
  return str(i-1)


def to_id(s):
  """Covert text to ids."""
  if s == "+": return 11
  if s == "*": return 12
  return int(s) + 1

def get_batch(max_length, batch_size, do_train, task, offset=None, preset=None):
  """Get a batch of data, training or testing."""
  inputs = []
  targets = []
  length = max_length
  if preset is None:
    cur_set = test_set[task]
    counters = test_counters
    if do_train:
      cur_set = train_set[task]
      counters = train_counters
    while not cur_set[length]:
      length -= 1
  for b in range(batch_size):
    if preset is None:
      cur_ind = counters[length]
      elem = cur_set[length][cur_ind]
      cur_ind += 1
      if cur_ind >= len(cur_set[length]):
        random.shuffle(cur_set[length])
        cur_ind=0
      counters[length]=cur_ind
      if offset is not None and offset + b < len(cur_set[length]):
        elem = cur_set[length][offset + b]
    else:
      elem = preset
    inp, target = elem[0], elem[1]
    assert len(inp) <= length
    inputs.append(inp + [0 for l in range(max_length - len(inp))])
    targets.append(target + [0 for l in range(max_length - len(target))])
  new_input = inputs
  new_target = targets
  return new_input, new_target


def print_out(s, newline=True):
  """Print a message out and log it to file."""
  if log_filename:
    try:
      with gfile.GFile(log_filename, mode="a") as f:
        f.write(s + ("\n" if newline else ""))
    # pylint: disable=bare-except
    except:
      sys.stdout.write("Error appending to %s\n" % log_filename)
  sys.stdout.write(s + ("\n" if newline else ""))
  sys.stdout.flush()


def accuracy(inpt, output, target, batch_size, nprint):
  """Calculate output accuracy given target."""
  assert nprint < batch_size + 1
  def task_print(inp, output, target):
    stop_bound = 0
    print_len = len(inp)
    #while print_len < len(target) and target[print_len] > stop_bound:
    #  print_len += 1
    print_out("    i: " + " ".join([str(i) for i in inp]))
    print_out("    o: " +
              " ".join([str(output[l]) for l in range(print_len)]))
    print_out("    t: " +
              " ".join([str(target[l]) for l in range(print_len)]))
  decoded_target = target
  decoded_output = output
  total = 0
  errors = 0
  seq = [0 for b in range(batch_size)]
  for l in range(len(decoded_output[0])):
    for b in range(batch_size):
      #if decoded_target[b][l] > 0:
      total += 1
      if decoded_output[b][l] != decoded_target[b][l]:
        seq[b] = 1
        errors += 1
  e = 0  # Previous error index
  for _ in range(min(nprint, sum(seq))):
    while seq[e] == 0:
      e += 1
    task_print(inpt[e],decoded_output[e],decoded_target[e])
    e += 1
  # for b in range(nprint - errors):
  #   task_print(inpt[b], decoded_output[b], decoded_target[b])
  return errors, total, sum(seq)


def safe_exp(x):
  perp = 10000
  if x < 100: perp = math.exp(x)
  if perp > 10000: return 10000
  return perp
