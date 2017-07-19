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

"""AdaMax optimizer
It is well suited for recurrent networks where exploding gradients is an issue.  
AdaMax has a desirable property that variable updates do not exceed the learning rate. 
In addition, gradient clipping proportional to the accumulated past gradients is introduced, 
eliminating the need for a pre-determined clipping threshold.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

    Gradient clipping is introduced in [Freivalds et. al., 2017](https://arxiv.org/abs/1702.08727)
    ([pdf](https://arxiv.org/pdf/1702.08727)).   

    This class provides lazy handling of gradient updates for sparse variables.
    It only updates moving-average accumulators for sparse variable indices that
    appear in the current batch, rather than updating the accumulators for all
    indices. 
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_gradients = True, clip_multiplier=1.2, clip_epsilon=1e-4, use_locking=False, name="Adamax"):
        """Construct a new AdaMax optimizer.
        
        Args:
            learning_rate: A Tensor or a floating point value. The learning rate.
            beta1: A float value or a constant float tensor.
              The exponential decay rate for the 1st moment estimates.
            beta2: A float value or a constant float tensor.
              The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            clip_gradients: Whether to perform gradient clipping
            clip_multiplier: Multiplier for second moment estimate for gradient clipping. Should be >1. Large values correspond to less clipping.
            clip_epsilon: Gradients smaller than this are not clipped. Also, it has some effect on the first few optimization steps.
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
        self.clip_gradients = clip_gradients

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self.clip_multiplier_t = None
        self.clip_epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
        self.clip_multiplier_t = ops.convert_to_tensor(self.clip_multiplier, name="clip_multiplier")
        self.clip_epsilon_t = ops.convert_to_tensor(self.clip_epsilon, name="clip_epsilon")

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
        clip_multiplier_t = math_ops.cast(self.clip_multiplier_t, var.dtype.base_dtype)
        clip_epsilon_t = math_ops.cast(self.clip_epsilon_t, var.dtype.base_dtype)

        v = self.get_slot(var, "v")
        # clip gradient so that each value exceeds its previous maximum by no more than clip_multiplier
        if self.clip_gradients:
            clipVal = v * clip_multiplier_t + clip_epsilon_t
            grad = clip_ops.clip_by_value(grad, -clipVal, clipVal)

        # m := beta1 * m + (1 - beta1) * g_t

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + (1. - beta1_t) * grad, use_locking=self._use_locking)
        # v := max(beta2 * v , abs(grad))
        v_t = state_ops.assign(v,math_ops.maximum(beta2_t * v, math_ops.abs(grad)), use_locking=self._use_locking)
        # variable -= learning_rate * m_t / (epsilon_t + v_t)
        # we do not use bias-correction term for the first moment; it does not give observable benefit
        var_update = state_ops.assign_sub(var, lr_t * m_t / (v_t+epsilon_t), use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, v_t, m_t])


    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        clip_multiplier_t = math_ops.cast(self.clip_multiplier_t, var.dtype.base_dtype)
        clip_epsilon_t = math_ops.cast(self.clip_epsilon_t, var.dtype.base_dtype)

        v = self.get_slot(var, "v")
        v_slice = array_ops.gather(v, grad.indices)

        #clip gradient so that each value exceeds its previous maximum by no more than clip_multiplier
        clipped_values = grad.values
        if self.clip_gradients:
            clipVal = v_slice * clip_multiplier_t + clip_epsilon_t
            clipped_values = clip_ops.clip_by_value(grad.values, -clipVal, clipVal)

        # m := beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t_values = beta1_t * array_ops.gather(m, grad.indices) + (1 - beta1_t) * clipped_values
        m_t = state_ops.scatter_update(m, grad.indices, m_t_values, use_locking=self._use_locking)

        # v := max(beta2 * v , abs(grad))
        v_t_values = math_ops.maximum(beta2_t * v_slice, math_ops.abs(clipped_values))
        v_t = state_ops.scatter_update(v, grad.indices, v_t_values, use_locking=self._use_locking)

        # variable -= learning_rate * m_t / (epsilon_t + v_t)
        # we do not use bias-correction term for the first moment; it does not give observable benefit
        var_update = state_ops.scatter_sub(var, grad.indices,
                                           lr_t * m_t_values / (v_t_values + epsilon_t),
                                           use_locking=self._use_locking)
        return control_flow_ops.group(var_update, v_t, m_t)
