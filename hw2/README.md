# 前情提要

我们已经在 `hw1` 中实现了一个基础的机器学习框架。 我们来看一些潜在的问题:

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



# 开始 hw2!

我们的机器学习框架的模块化架构是这样的:

![](https://notes.sjtu.edu.cn/uploads/upload_530ef2e28bd2d1e0a58fe950a6eeda33.png)

## initialization

首先是写几个初始化 Tensor 的函数， 如

```python
def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"

    std = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, mean = 0, std = std, **kwargs)
```

这里用到的 `randn` 表示正态分布， 来自 `init/init_basic.py`:

```python
def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random normal with specified mean and std deviation"""
    device = ndl.cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
```

