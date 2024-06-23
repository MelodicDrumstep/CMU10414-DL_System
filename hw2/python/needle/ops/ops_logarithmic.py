from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z_to_be_broadcasted = array_api.max(Z, self.axes, keepdims = True)
        # This compute the maximum of Z along with the specified axes. And it keeps
        # the dimension in order to be broadcasted when subtract by Z.

        max_Z_to_be_added = array_api.max(Z, self.axes, keepdims = False)
        # This compute the maximum of Z along with the specified axes. And it does not keep
        # the dimension in order to be added to the result

        return array_api.log(array_api.sum(array_api.exp(Z - max_Z_to_be_broadcasted), self.axes)) + max_Z_to_be_added
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_Z_to_be_broadcasted = Z.realize_cached_data().max(self.axes, keepdims=True)
        # This compute the maximum of Z along with the specified axes. And it keeps
        # the dimension in order to be broadcasted when subtract by Z.
        
        Z_after_subtracting_max = exp(Z - max_Z_to_be_broadcasted)
        Z_sum = summation(Z_after_subtracting_max, self.axes)
        grad_of_log_part = out_grad / Z_sum
        # compute log(M)' = 1 / M

        grad_of_log_part_after_dimension_adjustment = grad_of_log_part.reshape(max_Z_to_be_broadcasted.shape).broadcast_to(Z.shape)
        # adjust the dimension

        return grad_of_log_part_after_dimension_adjustment * Z_after_subtracting_max
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

