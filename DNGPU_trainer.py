# The MIT License (MIT)
#
# Copyright (c) 2016 Institute of Mathematics and Computer Science, Latvia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Improving the Neural GPU Architecture for Algorithm Learning"""

import os

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import time

from DNGPU_model import DNGPU
import random
import data_utils_2 as data_gen

# common settings
dropout_keep_prob = 0.9
training_iters = 20000
display_step = 200
max_test_length = 5000
batchSize = 32
test_data_size = 1024
data_size = 10000
data_gen.bins = [8, 12, 16, 20, 25, 28, 31, 36, 41]


# suggested settings for binary multiplication
task = "bmul"
n_input = 13 #range of input digits
n_output = 3 #range of output digits
n_hidden = 48*2 # number of maps

# suggested settings for binary addition
# task = "badd"
# n_input = 13 #range of input digits
# n_output = 3 #range of output digits
# n_hidden = 48 # number of maps
# dropout_keep_prob = 0.5

# suggested settings for base-4 multiplication
# task = "qmul"
# n_input = 13
# n_output = 5
# n_hidden = 48*4 # number of maps


# suggested settings for decimal multiplication with binary encoding
# task = "mulbcd"
# n_input = 13
# n_output = 5
# data_gen.bins = [9,17,25,33,41] # we have to specify different bins here since for many lengths we have no examples to train
# n_hidden = 48*4 # number of maps

# suggested settings for sorting numbers in range 0 to 5
# task = "sort"
# n_input = 6
# n_output = 6
# n_hidden = 48 # number of maps


model_file = "/tmp/varWeights.ckpt"
image_path = "/tmp/images"
countList = [batchSize for x in data_gen.bins]
np.set_printoptions(linewidth=2000)
# seed = 125459
# tf.set_random_seed(seed)
# random.seed(seed)
# np.random.seed(seed)

#prepare training and test data
max_length = data_gen.bins[-1]
for l in xrange(1,max_length+1):
    data_gen.init_data(task, l, data_size, n_input)
data_gen.collectBins()
data_gen.init_data(task, data_gen.forward_max, test_data_size, n_input)


# generates training data
def genTrainingData(forTraining):
    x=[]
    y=[]
    for i in xrange(len(data_gen.bins)):
        seq_len = data_gen.bins[i]
        seq_count=countList[i]
        data, labels = data_gen.get_batch(seq_len,seq_count,forTraining, task)
        x+=[data]
        y+=[labels]

    return x,y


# generates test data
def genTestData(length,count):
    data, labels = data_gen.get_batch(length, count, False, task)
    return [data], [labels]

# creates images of execution trace on a random input
# It requires a model snapshot written to file
def showPicture(test_length, path):
    if not os.path.exists(path):
        os.makedirs(path)

    data_gen.init_data(task, test_length, 1, n_input)
    while len(data_gen.train_set[task][test_length])==0:
        test_length += 1
        data_gen.init_data(task, test_length, 1, n_input)
    data_gen.resetCounters()

    with tf.Graph().as_default(),tf.device('/cpu:0'):
        tester = DNGPU(n_hidden, [test_length], n_input, [1], n_output, dropout_keep_prob)
        tester.createTestGraph(test_length)
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, model_file)
            if not os.path.exists(path): os.makedirs(path)

            batch_xs, batch_ys = genTestData(test_length, 1)
            print batch_xs, batch_ys
            mem = tester.getAllMem(sess, batch_xs, batch_ys)
            mem = np.squeeze(mem, 1)

            width = mem.shape[1]
            height = mem.shape[0]
            for unit in range(n_hidden):
                img=np.zeros((height,width),dtype=np.float32)
                for x in range(width):
                    for y in range(height):
                        img[y,x]=mem[y, x,unit]

                mpimg.imsave(path+"/frame"+str(unit)+".png",img, cmap='gray')

#Perform training
with tf.Graph().as_default():
    learner = DNGPU(n_hidden, data_gen.bins, n_input, countList, n_output, dropout_keep_prob)
    learner.createGraph()
    tf.get_variable_scope().reuse_variables()
    learner.createTestGraph(data_gen.forward_max)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #saver.restore(sess, model_file)

        batch_xs, batch_ys = genTrainingData(False)
        step = 1
        loss=0
        avgLoss=0
        avgRegul=0
        acc=1
        prev_loss = [1000000]*4
        start_time = time.time()
        batch_xs_long, batch_ys_long = genTestData(data_gen.forward_max, batchSize)
        long_accuracy, _ = learner.getAccuracy(sess, batch_xs_long, batch_ys_long)
        print "Iter", 0, "time =", 0,
        print "accuracy on length", data_gen.forward_max, "=", long_accuracy

        while step < training_iters:
            if step % display_step == 0:
                avgLoss/=display_step
                avgRegul/=display_step
                step_time = time.time() - start_time
                #start_time = time.time()
                lr = learner.getLearningRate(sess)
                print "Iter", step, "time =",step_time,"lr =",lr, 'max_loss =', loss, 'min_accuracy =', acc,'avgLoss =', avgLoss,

                batch_xs_long, batch_ys_long = genTestData(data_gen.forward_max, batchSize)
                long_accuracy, _=learner.getAccuracy(sess, batch_xs_long, batch_ys_long)
                print "accuracy on length",data_gen.forward_max,"=", long_accuracy

                # set saturation weight proportional to average loss
                learner.set_saturation_weight(sess, avgLoss / (avgRegul + 1e-20))

                # decrease learning rate if no progress is made in 3 checkpoints
                prev_loss.append(avgLoss)
                if min(prev_loss[-2:]) > min(prev_loss[-3:]):
                    sess.run(learner.lr_decay_op)
                loss = 0
                acc = 1
                avgLoss = 0
                avgRegul = 0

            batch_xs, batch_ys = genTrainingData(True)
            loss1, acc1, perItemCost, costList, regul1 = learner.train(sess, batch_xs, batch_ys, 1)
            avgLoss+=loss1
            avgRegul+=regul1

            loss = max(loss, loss1)
            acc = min(acc, acc1)
            step += 1

        print "Optimization Finished!"
        saver.save(sess, model_file)

# create execution trace images
showPicture(200, image_path)

# test generalization to longer examples
test_length = 7
test_examples=1024

while test_length<max_test_length:
    if test_length>4001: test_length = 4001
    if len(data_gen.test_set[task][test_length]) == 0:
        data_gen.init_data(task, test_length, test_data_size, n_input)
    while len(data_gen.test_set[task][test_length])==0:
        test_length += 1
        data_gen.init_data(task, test_length, test_data_size, n_input)

    data_gen.resetCounters()
    batchSize = 1
    if test_length < 2000: batchSize = 16
    if test_length < 800: batchSize = 128

    with tf.Graph().as_default():#, tf.device('/cpu:0'):
        tester = DNGPU(n_hidden, [test_length], n_input, [batchSize], n_output, dropout_keep_prob)
        tester.createTestGraph(test_length)
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, model_file)
            errors, seq_errors,total = 0, 0, 0
            for iter in range(test_examples/batchSize):
                batch_xs, batch_ys = genTestData(test_length, batchSize)
                acc1, test_result=tester.getAccuracy(sess, batch_xs, batch_ys)
                er, tot,seq_er=  data_gen.accuracy(batch_xs[0], test_result, batch_ys[0], batchSize, 0)
                errors+=er
                seq_errors+=seq_er
                total+=tot

            acc_real = 1.0-float(errors)/total
            print "Testing length:", test_length, "accuracy=", acc_real, "errors =", errors, "incorrect sequences=",seq_errors
    test_length=test_length*4/3



