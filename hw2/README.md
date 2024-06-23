# 前情提要

我们已经在 `hw1` 中实现了一个基础的深度学习框架。 我们来看一些潜在的问题:

## memory bug?

如果我写出这样的代码

```python
grad = ndl.Tensor([1, 1, 1], dtype="float32")
lr = 0.1
for i in range(5):
    w = w + (-lr) * grad
```

我会发现这里 `w = w + (-lr) * grad` 会生成多个计算图结点， 这些计算图结点导致之前的结点仍然有引用计数从而无法释放。 这样写最终会导致内存爆满。 也就是说， 我们希望只使用 `w` 的值， 而不要为这个计算图链接上新的结点。

所以，如果我只想用 `w` 的值， 我可以定义这样的 `method`:

```python
    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data())

    # below defines a data member of the class
    # on our own
    @property
    def data(self):
        return self.detach()
```

所以我们可以用 `w.data` 创建一个不包含其他计算图结点指针的, 孤立的, 存着 `w` 值的结点。

我们可以这样来实现之前的需求:

```python
grad = ndl.Tensor([1, 1, 1], dtype="float32")
lr = 0.1
for i in range(5):
    w.data = w.data + (-lr) * grad.data
```

```
Tips: python 中赋值语句总是创建对象的引用值， 而不是深拷贝。 所以我们的 w.data 实际上没有深拷贝数据区， 而是共享指向数据区的指针。 所以上述直接 `w.data = w.data + (-lr) * grad.data` 改到了原始 `w` 的数据。
```

另外， 可以看我们实现的 `make_from_op` method:

```python
    # this method will create a Tensor node out of a certain operation
    # and inputs
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor
```

当我们创建一个新的结点的时候， 如果这个结点不需要计算梯度， 且 `input` 也都不需要计算梯度， 那我们认为没必要存之前结点的引用了， 直接创建一个 `tensor.detach()`， 即孤立的的结点。 这样可以节省资源， 避免持有引用导致资源无法释放。

## 数据精度

对于 `softmax`: 

$\begin{equation}
 z_i = \frac{exp(x_i)}{\sum_k exp(x_k)}
\end{equation}$

如果我们传入的 `x_i` 很大， 那么很容易发生浮点数溢出。 我们如何解决这个问题呢？

我们可以发现这个性质:

$\begin{equation}
 z_i = \frac{exp(x_i)}{\sum_k exp(x_k)} = \frac{exp(x_i-c)}{\sum_k exp(x_k-c)}
\end{equation}$

所以我把每个 $x_i$ 都减去 $max(x_i)$ 好了。

可以实现成这样:

```python
x = np.array([1000, 10000, 100], dtype="float32")
def softmax_stable(x):
    x = x - np.max(x)
    z = np.exp(x)
    return z / np.sum(z)

softmax_stable(x)
```

## 质疑 pytorch， 理解 pytorch, 成为 pytorch

我们在 hw1 里写的深度学习框架有什么问题？ 问题在于， 模块化做得还不够好。 神经网络从研究者的角度来看是分层、模块化的。 我们希望解耦各个模块， 从而使得研究者不用关心各个模块的实现细节。

如今 pytorch 已经成为了最流行的深度学习框架， 它最成功的地方便是模块化， 使得上手非常容易， 也符合人们对于新事物的认知习惯。 有时间了我一定要去读一读 pytorch 的源码: https://github.com/pytorch/pytorch.


# 开始 hw2!

我们的深度学习框架的模块化架构是这样的:

![](https://notes.sjtu.edu.cn/uploads/upload_530ef2e28bd2d1e0a58fe950a6eeda33.png)

我们一步步来完善这个框架。

## initialization

首先来写初始化部分。 

先写几个初始化 Tensor 的函数. 从 10414 的课堂上我们已经知道， 深度学习的初始化还真不是一件无足轻重的事情， 有时候初始化选取的不好， 又没有其他补救措施(如 normalization) 的话， 可能就无法训练下去了。 我们来写这几个经过检验好用的初始化函数, 请注意， 这些函数的返回类型都是 `Tensor`。

### Xavier uniform

它的公式是这样的:

从 $\mathcal{U}(-a, a)$ 中采样， 其中

$a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$

其中参数是:

+ gain : 可选的配置参数

+ fan_in : 输入维度

+ fan_out : 输出维度

那就直接实现成这样:

```python
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low = -a, high = a, **kwargs)
```

`python` 在传参时， 可以在函数原型中用 `*args` 表示接收任意数量的非关键字参数, 并当作一个 `tuple` 传入; `**kwargs` 表示接收任意数量的关键字参数， 并当作一个 `dict` 传入。

这里用到的 `rand` 表示均匀分布， 来自 `init/init_basic.py`:

```python
def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random numbers uniform between low and high"""
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
```

### Xavier normal

这个初始化方案就是把上一个方案的概率分布从均匀分布改成了正态分布。直接实现成这样:

```python
def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mea = 0, std = std, **kwargs)
```

这里用到的 `randn` 表示正态分布， 来自 `init/init_basic.py`:

```python
def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random normal with specified mean and std deviation"""
    device = ndl.cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
```

### kaiming_uniform

它的公式是这样:

从 $\mathcal{U}(-\text{bound}, \text{bound})$ 中采样， 其中

$\begin{equation}
\text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_in}}}
\end{equation}$

推荐的 `gain` 值 : $\text{gain}=\sqrt{2}$.

所以直接实现成这样:

```python
def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(6 / fan_in)
    return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs)
```

### kaiming_normal

这个初始化方案就是把上一个方案的概率分布从均匀分布改成了正态分布。直接实现成这样:

```python
def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    std = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, mean = 0, std = std, **kwargs)
```

## nn.Module

接下来我们来实现 `nn.Module` 部分， 也是深度学习框架的主体部分。我们先来阅读一些代码。

### Parameter

我们把 `Paramater` 类设计成了一个 `Tensor` 的派生类:

```python
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
```
 
### Module 基类

```python
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
```

我们对 `Module` 的实现很简单， 实现了递归获取所有参数的函数和递归获取所有子 `Module` 的函数。

### Identity

一个很简单的例子是 `Identity Module`, 直接实现成这样:

```python
class Identity(Module):
    def forward(self, x):
        return x
```

因为 `Module` 是建立于 `Tensor / TensorOp` 的抽象之上的， 所以我们可以只实现 `forward` 函数，而不实现 `backward` 函数， `backward` 函数可以直接使用各 `TensorOp` 的 `backward` 函数。 但是某些 `Module`（如卷积层） 也会需要我们定制特定算子， 实现更加高效的的 `backward` 函数， 用于提升性能。  

接下来的部分就是需要我们动手去写的了。

### Linear

这个就是一个简单的线性层。

```python
class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # use kaiming uniform to initialize the parameters
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad = True))
        # if bias = False, let self.bias be None
        if bias == True:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad = True))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x @ A^T + b
        xAT = X.matmul(self.weight)
        if self.bias != None:
            xAT += self.bias.broadcast_to(xAT.shape)
        return xAT
        ### END YOUR SOLUTION
```

这里任务书中指出, `bias` 的初始化要用 `fan_in = out_features`。 具体原因我并没有太理解， 但是这样维度是不会有问题的， 因为最后前向传播计算时， `broadcast` 会将 `bias` 向量维度调整为 `(N, out_features)`.

### ReLU

```python
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return needle.ops.relu(x)
        ### END YOUR SOLUTION
```

`ReLU Module` 不含参数， 直接用已经定义过的 `ReLU Op` 算子即可。

### Sequential

及多个 `Module` 的线性连接， 实现很简单。

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
            output = module(output)
        return output
        ### END YOUR SOLUTION
```

记得不要直接改输入的 `x`。 `python` 中都是传引用而不是传值， 这样会改到外侧的 `x`.

### LogSumExp

这个任务又是写算子了, 是为了 `softmax module` 服务的。 是写这个公式的算子:

$$\text{LogSumExp}(z) = \log (\sum_{i} \exp (z_i - \max{z})) + \max{z}$$

首先是前向传播， 前向传播直接用 `array_api` 的函数就行了， 和梯度无关。

反向传播的部分使用链式法则即可。

算子实现是这样:

```python
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
        grad_of_log_part_after_dimension_adjustment = grad_of_log_part.reshape(max_Z_to_be_broadcasted.shape).broadcast_to(Z.shape)
        return grad_of_log_part * Z_after_subtracting_max
        ### END YOUR SOLUTION
```