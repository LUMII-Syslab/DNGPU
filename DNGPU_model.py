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

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from adamax import AdamaxOptimizer
import numpy as np

class DNGPU:
    def __init__(self, num_units, bins, n_input, count_list, n_classes, dropout_keep_prob):
        self.n_classes = n_classes
        self.n_input = n_input
        self.num_units = num_units
        self.bins = bins
        self.count_list = count_list
        self.is_training = tf.placeholder(tf.int32, shape=[], name="is_training")
        self.accuracy = None
        self.test_accuracy = None
        self.base_cost = None
        self.sat_loss = None
        self.gnorm = None
        self.optimizer = None
        self.cost_list = None
        self.global_step = tf.Variable(0, trainable=False)
        initial_learning_rate = 0.01*48/self.num_units
        min_learning_rate = initial_learning_rate/60
        self.learning_rate = tf.Variable(initial_learning_rate, trainable=False)
        self.beta2_rate = tf.maximum(0.0005,tf.train.exponential_decay(0.01, self.global_step, 2000, 0.5, staircase=False))
        self.bin_losses = []
        self.dropout_keep_prob = dropout_keep_prob
        self.allMem = None
        self.saturation_costs=[] # here we will collect all saturation costs
        self.saturation_limit = 0.9
        self.tmpfloat = tf.placeholder("float")
        self.saturation_weight = tf.Variable(1e-3, trainable=False)
        self.assign_saturation_weight_op = self.saturation_weight.assign(self.tmpfloat)
        self.x_input = []
        self.y_input = []
        self.test_x = None
        self.test_y = None
        self.lr_decay_op = self.learning_rate.assign(tf.maximum(min_learning_rate,self.learning_rate * 0.7))
        self.result = None
        self.shift_filter = self.create_shift_filter()

    def hard_sigmoid(self, x_in, weight=None):
        self.add_saturation_cost(x_in, weight)
        x= x_in*0.5+0.5
        cut_bottom = tf.nn.relu(x)
        cut_both =tf.minimum(1.0,cut_bottom)
        return cut_both

    def hard_tanh(self, x, weight = None):
        self.add_saturation_cost(x, weight)
        cut_bottom = tf.maximum(x,-1.0)
        cut_both =tf.minimum(1.0,cut_bottom)
        return cut_both

    def dropout(self, d):
        d = tf.cond(self.is_training > 0, lambda: tf.nn.dropout(d, self.dropout_keep_prob), lambda: d)
        return d

    def add_saturation_cost(self, var, weight=None):
        """Calculate saturation cost."""
        sat_loss = tf.nn.relu(tf.abs(var) - self.saturation_limit)
        cost = tf.reduce_sum(sat_loss)
        if weight:cost*=weight
        self.saturation_costs.append(cost)

    def conv_linear(self, input, kernel_width, nin, nout, bias_start, prefix):
        """Convolutional linear map."""

        with tf.variable_scope(prefix):
            filter = tf.get_variable("CvK", [kernel_width, nin, nout])
            res = tf.nn.conv1d(input, filter, 1, "SAME")
            bias_term = tf.get_variable("CvB", [nout], initializer=tf.constant_initializer(0.0))
            return res + bias_term + bias_start

    def create_shift_filter(self):
        # prepare convolution filter that performs shifting
        #input is of shape [batch, 1, in_width, in_channels]
        #shiftFilter is of shape [1, filter_width, in_channels, 1]

        shifted_maps = self.num_units/3
        baseFilter = [[0,1,0]]*(self.num_units-2*shifted_maps)+[[1,0,0]]*shifted_maps+[[0,0,1]]*shifted_maps
        shiftFilter = tf.constant(np.transpose(baseFilter), dtype=tf.float32)
        shiftFilter = tf.expand_dims(tf.expand_dims(shiftFilter,0), 3)
        return shiftFilter


    def DCGRU(self, mem, kernel_width, prefix):
        """Convolutional diagonal GRU."""

        def conv_lin(input, suffix, bias_start):
            return self.conv_linear(input, kernel_width, self.num_units, self.num_units, bias_start,prefix + "/" + suffix)

        # perform shift
        mem_shifted = tf.squeeze(tf.nn.depthwise_conv2d(tf.expand_dims(mem,1), self.shift_filter,[1,1,1,1],'SAME'),[1])

        # calculate the new value
        reset = self.hard_sigmoid(conv_lin(mem, "r", 0.5))
        candidate = self.hard_tanh(conv_lin(reset * mem, "c", 0.0))
        gate = self.hard_sigmoid(conv_lin(mem, "g", 0.7))
        candidate =self.dropout(candidate)

        candidate =  gate*mem_shifted + (1 - gate)*candidate
        return candidate


    def createLoss(self, x_in_indices, y_in, length):
        """perform loss calculation for one bin """

        # create mask
        mask_1 = tf.cast(tf.equal(x_in_indices, 0), tf.float32)
        mask_2 = tf.cast(tf.equal(y_in, 0), tf.float32)
        mask = tf.stack([1.0-mask_1*mask_2]*self.num_units,axis=2)

        # the input layer
        x_in = tf.one_hot(x_in_indices, self.n_input, dtype=tf.float32)
        cur = self.conv_linear(x_in, 1, self.n_input, self.num_units, 0.0, "input")
        cur = self.hard_tanh(cur, length)
        cur = self.dropout(cur)
        cur*=mask

        allMem = [cur] #execution trace

        #computation steps
        with vs.variable_scope("steps") as gruScope:
            for i in range(length):
                cur = self.DCGRU(cur, 3, "dcgru")
                cur *= mask
                allMem.append(cur)
                gruScope.reuse_variables()

        # output layer and loss
        allMem_tensor = tf.stack(allMem)
        prediction = self.conv_linear(cur, 1, self.num_units, self.n_classes, 0.0, "output")
        costVector = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = y_in)  # Softmax loss

        result = tf.argmax(prediction, 2)
        correct_pred = tf.equal(result, y_in)
        perItemCost = tf.reduce_mean(costVector, (1))
        cost = tf.reduce_mean(perItemCost)

        correct_pred = tf.cast(correct_pred, tf.float32)
        accuracy = tf.reduce_mean(correct_pred)

        return cost, accuracy, allMem_tensor, prediction, perItemCost, result

    def createTestGraph(self, test_length):
        """Creates graph for accuracy evaluation"""
        with vs.variable_scope("var_lengths"):
            itemCount = self.count_list[0]
            self.test_x = tf.placeholder("int32", [itemCount, test_length])
            self.test_y = tf.placeholder("int64", [itemCount, test_length])
            _, self.test_accuracy, self.allMem, _, _, self.result = self.createLoss(self.test_x,self.test_y,test_length)

    def createGraph(self):
        """Creates graph for training"""
        self.base_cost=0.0
        self.accuracy = 0
        num_sizes = len(self.bins)
        self.cost_list = []
        sum_weight=0
        self.bin_losses = []
        saturation_loss = []

        # Create all bins and calculate losses for them

        with vs.variable_scope("var_lengths"):
            for seqLength,itemCount, ind in zip(self.bins, self.count_list, xrange(num_sizes)):
                x_in = tf.placeholder("int32", [itemCount, seqLength])
                y_in = tf.placeholder("int64", [itemCount, seqLength])
                self.x_input.append(x_in)
                self.y_input.append(y_in)
                self.saturation_costs = []
                c, a, _, _, perItemCost, _ = self.createLoss(x_in,y_in,seqLength)

                weight = 1.0#/seqLength
                sat_cost = tf.add_n(self.saturation_costs) / ((seqLength ** 2) * itemCount)
                saturation_loss.append(sat_cost*weight)
                self.bin_losses.append(perItemCost)
                self.base_cost += c * weight
                sum_weight+=weight
                self.accuracy += a
                self.cost_list.append(c)
                tf.get_variable_scope().reuse_variables()

        # calculate the total loss
        self.base_cost /= sum_weight
        self.accuracy /= num_sizes

        self.sat_loss = tf.reduce_sum(tf.stack(saturation_loss))*self.saturation_weight / sum_weight
        cost = self.base_cost + self.sat_loss

        # add gradient noise proportional to learning rate
        tvars = tf.trainable_variables()
        grads_0 = tf.gradients(cost, tvars)

        grads = []
        for grad in grads_0:
                grad1 = grad+tf.truncated_normal(tf.shape(grad)) * self.learning_rate*1e-4
                grads.append(grad1)

        # optimizer
        optimizer = AdamaxOptimizer(self.learning_rate, beta1=0.9, beta2 = 1.0-self.beta2_rate, epsilon=1e-8)
        self.optimizer = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        # some values for printout
        max_vals=[]

        for var in tvars:
            varV = optimizer.get_slot(var, "m")
            max_vals.append(varV)

        self.gnorm = tf.global_norm(max_vals)
        self.cost_list = tf.stack(self.cost_list)

    def prepare_dict(self, batch_xs_list, batch_ys_list, is_training):
        """Prepares a dictionary of input output values for all bins to do training"""
        feed_dict = {self.is_training: is_training}
        for x_in, data_x in zip(self.x_input, batch_xs_list):
            feed_dict[x_in.name] = data_x
        for y_in, data_y in zip(self.y_input, batch_ys_list):
            feed_dict[y_in.name] = data_y

        return feed_dict

    def prepare_test_dict(self, batch_xs_list, batch_ys_list):
        """Prepares a dictionary of input output values for all bins to do testing"""
        feed_dict = {self.is_training: 0}
        feed_dict[self.test_x.name] = batch_xs_list[0]
        feed_dict[self.test_y.name] = batch_ys_list[0]
        return feed_dict

    def getAllMem(self, sess,batch_xs_list, batch_ys_list):
        """Gets an execution trace for the given inputs"""
        feed_dict = self.prepare_test_dict(batch_xs_list, batch_ys_list)
        mem = sess.run((self.allMem), feed_dict=feed_dict)
        return mem


    def getAccuracy(self, sess,batch_xs_list, batch_ys_list):
        """Gets accuracy on the given test examples"""
        feed_dict = self.prepare_test_dict(batch_xs_list, batch_ys_list)
        acc, result = sess.run((self.test_accuracy, self.result), feed_dict=feed_dict)
        return acc, result

    def getLearningRate(self, sess):
        rate = sess.run((self.learning_rate))
        return rate

    def printLoss(self, sess,batch_xs_list, batch_ys_list):
        """prints training loss on the given inputs"""
        feed_dict = self.prepare_dict(batch_xs_list, batch_ys_list, 0)
        acc, loss, costs, norm11, regul, beta2 = sess.run((self.accuracy, self.base_cost, self.cost_list, self.gnorm, self.sat_loss, self.beta2_rate),
                                                          feed_dict=feed_dict)
        print "Loss= " + "{:.6f}".format(loss) + \
              ", Accuracy= " + "{:.6f}".format(acc), norm11, beta2,
        return loss

    def train(self, sess, batch_xs_list, batch_ys_list, do_dropout=1):
        """do training"""
        feed_dict = self.prepare_dict(batch_xs_list, batch_ys_list, do_dropout)

        res = sess.run([self.base_cost, self.optimizer, self.accuracy, self.cost_list, self.sat_loss] + self.bin_losses,
                       feed_dict=feed_dict)
        loss = res[0]
        acc=res[2]
        costs=res[3]
        regul = res[4]
        lossPerItem = res[5:]
        return loss, acc, lossPerItem, costs, regul


    def set_saturation_weight(self, sess, koef):
        curVal = sess.run(self.saturation_weight)
        curLearningRate = sess.run(self.learning_rate)
        koef *= curVal*curLearningRate
        koef = max(min(koef, 1e-3), 1e-20)
        sess.run(self.assign_saturation_weight_op, feed_dict={self.tmpfloat: koef})
