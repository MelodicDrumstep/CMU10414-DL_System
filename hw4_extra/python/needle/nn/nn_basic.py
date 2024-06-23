"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

# we define the Parameter class as a derived class of Tensor
class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

# This will return every parameter needed to compute this tensor
def _unpack_params(value: object) -> List[Tensor]:
    # if I'm a parameter, return myself
    if isinstance(value, Parameter):
        return [value]
    # if I'm a Module, call the parameter() method
    # to recursively return a list of paramter of every attribute of myself
    elif isinstance(value, Module):
        return value.parameters()
    # if I'm a dict, iterate all values in this dict and recursively call this method
    # this will happen when value is Module and call value.pamameters()
    # because self.__dict__ is a dict of all attributes
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    # if I'm a tuple, recursively call this function of every elements
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

# This method will find all Module child object
def _child_modules(value: object) -> List["Module"]:
    # if I'm already a Module, add myself to the list
    # and recursively call this function to the attributes
    # the extend method of list will add the element in another list into this list
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    # if I'm a dict, recursively call this function to all values in the dict
    # This can happen when I call this function of a Module and Module will call this function
    # of its attributes dict
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    # if I'm a tuple, recursively call this function to all elements
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []

# This is the base class of every Module
class Module:
    def __init__(self):
        # default mode is training
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        # self.__dict__ stores the instance's attributes as a dictiornary
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        # Disable training, do not need gradient
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        # Enable training, need gradient
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        # the forward function
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias = True, device = None, dtype = "float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad = True))
        self.bias_term = Parameter(init.kaiming_uniform(in_features, 1, requires_grad = True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = X.matmul(self.weight)
        if self.bias:
            output += self.bias_term.broadcast_to(output.shape)
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
