# This code is adapted from https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
# The MIT License (MIT)
#
# Original work Copyright (c) 2016 openai
# Modified work Copyright (c) 2016 Institute of Mathematics and Computer Science, Latvia
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

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm with gradient clipping.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,epsilon=1e-8,clip_multiplier=1.2, clip_epsilon = 1e-4, use_locking=False, name="Adamax"):
        """Construct a new AdaMax optimizer with gradient clipping.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability.
          clip_multiplier: Multiplier for second moment estimate for gradient clipping. We have not verified that the default value is optimal.
          clip_epsilon: Gradients smaller than this are not clipped. Also, it has some effect on the first few optimization steps. We have not verified that the default value is optimal.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adamax".
        """

        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self.clip_multiplier = clip_multiplier
        self.clip_epsilon = clip_epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        clipVal = m*self.clip_multiplier+self.clip_epsilon
        grad = tf.clip_by_value(grad, -clipVal, clipVal)
        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m_t = m.assign(tf.maximum(beta2_t * m, tf.abs(grad)))
        g_t = v_t / (m_t+epsilon_t)

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")