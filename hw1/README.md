这个 lab 是实现计算图的自动求导。

## 目标

我们希望实现怎样的机器学习框架？ 

我们希望用我们的机器学习框架 needle,  `softmax loss` 可以这样写:

```python
def softmax_loss(Z, I_y):
    return (ndl.log(ndl.exp(Z).sum((1,))).sum() - (I_y * Z).sum()) / Z.shape[0]
```

`nn_epoch` 可以这样写:

```python
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """
    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch
    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    for i in range(0, (y.size + batch - 1) // batch):
        x_batch = ndl.Tensor(X[i * batch : (i+1) * batch, :])
        y_batch = y[i * batch : (i+1) * batch]
        Z = ndl.relu(x_batch.matmul(W1)).matmul(W2)
        # Z = ReLU(X * W1) * W2
        I_y = np.zeros((batch, y.max() + 1))
        I_y[np.arange(batch), y_batch] = 1
        I_y = ndl.Tensor(I_y)
        # create I_y as numpy array and convert it to Tensor
        loss = softmax_loss(Z, I_y)
        # Create a loss node by applying softmax_loss function to Z and I_y
        loss.backward()
        # back propagate the gradients
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
        # update the weights
    return W1, W2
```

这里我简单画了一下这个计算图:

<img src="https://notes.sjtu.edu.cn/uploads/upload_40668cc7d5b325ada688b0c5a20a1051.jpg" width="500">

可以看到， 如此简单的一个算法， 计算图就已经很复杂了。 所以我觉得计算图肯定有很多优化点。

## Tensor

tensor 是机器学习中常用的名词， 可以理解为高维数组。

我们首先来看看 needle 框架如何定义 tensor 类:

首先， 我们定义 Value 类, 表示计算图中的一个值。 我为该类写了详细注释

```python
class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    # the operation with this value node
    inputs: List["Value"]
    # the inputs with this value node

    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        # if there's no operation with this value node
        # then it's the leaf node
        return self.op is None

    def __del__(self):
        # decrement the global counter  
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        # increment the global counter
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            # if this node does not require gradient
            # check if there's any input node that requires gradient
            # if so, I should mark this node as "requires_grad"
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    # classmethod is like statis member function in C++
    # it's bind to the whole class rather than some class objects
    # This method will create a constant value node
    # cls means the class itself
    # Why it's constant? Because this value node has no operation
    # and no input
    # How to use it? like this :
    # const_value = Value.make_const(data)
    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value


    # this class method will create a value node 
    # with a certain operation and inputs
    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            # if it's not in the lazy mode
            # compute the data if needed
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value
```

接下来是派生类 Tensor， 最核心的函数是 `backward`

```python
class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
        # kwargs means other parameters that are not defined
    ):
        if isinstance(array, Tensor):
            # if array is already a tensor
            # just copy the data
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            # if array is not a tensor
            # convert it to a tensor
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)
        # call the _init function defined in the base class
        # Value to initialize the node
        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    # static method is like static member function in C++
    # (It's more like static member function than "class method")
    # because it cannot access the class at all

    # this method will convert an array into a "array_api" array
    # to unified the data
    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

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

    # This method will create a const tensor node
    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor
    # ....

    # Attention!! This is the core function
    # backward will compute the gradient by topo order
    # see "compute_gradient_of_variables" for implementation
    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    # some operator overload here
    # ....
```

Tensor 是计算图中结点的类型。 

我们还需要定义运算的类型， 接下来看 `Op` 类。

## Op

首先是两个基类:

```python
class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
```

之后， 我们每个运算类都是派生自 `TensorOp` 的， 这部分代码写在 `ops_mathematic.py` 中。

如:

```python

# a + b
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)

class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        # d(x / y) / dx = 1 / y
        # d(x / y) / dy = -x / y^2
        return out_grad / node.inputs[1], -out_grad * node.inputs[0] / (node.inputs[1] ** 2)
```

我们自定义了 `compute` 函数， 用于正向计算， 和 `gradient` 函数， 计算梯度， 用于反向传播。 

## 前向计算 & 反向传播

这里我们需要明确几点:

+ 何时前向计算？

我们并不是每创建一个结点就自动通过运算类计算出值的， 而是第一次手动调用 `realize_cached_data` 的时候计算出值， 并将该值缓存， 下次需要时直接从缓存中取即可。

```python
    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data
```

注意到这里会递归调用 `realize_cached_data`， 所以如果刚构建好一个计算图， 第一次调用最终结点的 `realize_cached_data` 时， 就会调用整张图每个结点的 `op.compute` 计算出每个结点的值。

其实如果只是完成一次 `regression epoch`， 我们没必要用到 `loss` 的值，如

```python
    for i in range(0, (y.size + batch - 1) // batch):
        x_batch = ndl.Tensor(X[i * batch : (i+1) * batch, :])
        y_batch = y[i * batch : (i+1) * batch]
        Z = ndl.relu(x_batch.matmul(W1)).matmul(W2)
        # Z = ReLU(X * W1) * W2
        I_y = np.zeros((batch, y.max() + 1))
        I_y[np.arange(batch), y_batch] = 1
        I_y = ndl.Tensor(I_y)
        # create I_y as numpy array and convert it to Tensor
        loss = softmax_loss(Z, I_y)
        # Create a loss node by applying softmax_loss function to Z and I_y
        loss.backward()
        # back propagate the gradients
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
        # update the weights
    return W1, W2
```

这里我们需要计算出值的地方只有 ReLU 结点及之前。 (ReLU 的梯度需要结点值来计算)

但我们训练时通常也需要 `loss` 的值来判断收敛以及输出调试， 因此我们往往需要计算出 `loss` 的值, 如:

``` python
def loss_err(h, y):
    """Helper function to compute both loss and error"""
    I_y = np.zeros((y.shape[0], h.shape[-1]))
    I_y[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(I_y)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
```

这里的 `.numpy()` 函数会调用 `realize_cached_data` 并返回 `numpy` 数组。

+ 何时反向传播？

首先我们需要注意， 我们把 `grid` 定义为 `Tensor` 的成员， 它也是一个 `Tensor`.

```python
class Tensor(Value):
    # Notice that grad is a tensor!!!
    grad: "Tensor"
```

我们在反向传播时 `backward` 的参数也是一个 `Tensor`:

```python
# Attention!! This is the core function
# backward will compute the gradient by topo order
# see "compute_gradient_of_variables" for implementation
def backward(self, out_grad = None):
    out_grad = (
        out_grad
        if out_grad
        else init.ones(*self.shape, dtype=self.dtype, device=self.device)
    )
    compute_gradient_of_variables(self, out_grad)

def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )
```

这里我们最终结点的 `out_grad` 是全 1 的 `Tensor`， 即表示 $\frac{\partial y}{\partial y} = 1$.

下面我详细讲解一下这里的反向传播算法。

## back propogation

这里写的是 Reverse AD algorithm, 即下图中算法:

<img src="https://notes.sjtu.edu.cn/uploads/upload_c1cae9e85b59b478195ad95477261572.png" width="500">

实现如下:

```python
def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        # sum up the gradients from all output nodes
        grad = sum_node_list(node_to_output_grads_list[node])
        # fill in the fields of the node
        node.grad = grad
        # store the computed gradient in the "grad" field of this Tensor Node
        if node.op is None:
            continue
        # compute the gradients of the inputs
        input_grads = node.op.gradient_as_tuple(grad, node)
        for i, input_node in enumerate(node.inputs):
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            node_to_output_grads_list[input_node].append(input_grads[i])

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    is_visted = set()
    ans = []
    for node in node_list:
        topo_sort_dfs(node, is_visted, ans)
    return ans

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)
```

这里我们创建了一个 `map` : `node_to_output_grads_list`, 每次我们算到一个结点的偏导数， 就会把这个偏导数 `append` 到结点对应位置的 `list` 上。 轮到这个结点时， 只需要把它对应的 `list` 中的所有偏导数求和， 即为该节点的梯度， 然后将该 `gradient tensor` 存入结点的 `grad` 区域。

这里一定要理解 `gradient` 函数做了什么： 我们假设最终函数为 `L`, 该结点为 `y`, 该结点的输入为 `a, b ,c` 3 个 `Tensor`， 则 `gradient` 函数会输出一个 `list : ` $[\frac{\partial L}{\partial y}\frac{\partial y}{\partial a}, \frac{\partial L}{\partial y}\frac{\partial y}{\partial b}, \frac{\partial L}{\partial y}\frac{\partial y}{\partial c}]$.

所以我们如何定义每个 `Op` 的 `gradient` 函数？ 我们只需要知道用什么表达式计算 $\frac{\partial y}{\partial a}$ 就行了。

## 不显然的 gradient 函数


这里详细解释一下几个不显然的梯度是如何算的:

### ReLU

```python
class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(0, a)

    def gradient(self, out_grad, node):
        data = node.realize_cached_data().copy()
        data[data > 0] = 1
        return out_grad * Tensor(data)
```

这里直接取出 tensor node 存的数据(即 ReLU(...) 这个前向计算的时候算过的值)， 然后复制一份， 再把数据 $> 0$ 的部分设置为 $1$， ( $<= 0$ 的部分已经是 $0$ 了) 

### MatMul

首先看这部分讲解:

<img src="https://notes.sjtu.edu.cn/uploads/upload_19798b5f2d4122136aa020c0176105ac.png" width="400">
<img src="https://notes.sjtu.edu.cn/uploads/upload_5075b8f7dd0cd1421cdb594358ed2ae9.png" width="400">
<img src="https://notes.sjtu.edu.cn/uploads/upload_fcd5df3c3d2419db425c00cf47590656.png" width="400">
<img src="https://notes.sjtu.edu.cn/uploads/upload_ecc36d76780e29089df87892189995e9.png" width="400">
<img src="https://notes.sjtu.edu.cn/uploads/upload_d80e4b5528f6d6848481f0f1643ed019.png" width="400">
<img src="https://notes.sjtu.edu.cn/uploads/upload_16042b111e509192c833e9b2a94bc0c8.png" width="400">
<img src="https://notes.sjtu.edu.cn/uploads/upload_3c5f2b4ec8e4ea932524d68b3d27f21f.png" width="400">

矩阵乘法的规则是 :  若 $C = AB$, 则 $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}B^T$, $\frac{\partial L}{\partial B} = A^T\frac{\partial L}{\partial C}$.

按理说这里的 shape 是对应上的， 但是为了处理高维情况 (比如三维 Tensor 和二维 Tensor 相乘)， 这里进行了维度调整。

```python
# C = A x B
class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        # \partial L / \partial B = A^T * \partial L / \partial C
        # \partial L / \partial A = \partial L / \partial C * B^T
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
```

我们可以看不进行维度调整会出现什么问题。 比如， 对于样例

```python
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(6, 6, 5, 4)),
        ndl.Tensor(np.random.randn(4, 3)),
    )
```

这里 $A : 6 x 6 x 5 x 4$, $B : 4 x 3$, 则 $C : 6 x 6 x 5 x 3$.

因此我们假设 $outgrad : 6 x 6 x 5 x 3$.

那么 $lgrad = outgrad B^T : 6 x 6 x 5 x 4$. 和 $A$ 的形状匹配， 不需要调整了。

$rgrad = A^T outgrad : 6 x 6 x 4 x 3$. 和 $B$ 的形状不匹配。 如果放任不管， 那和 $B$ 算出来的其他梯度是无法相加的。 因此， 我们需要把它调整为 $4 x 3$.

如何调整？ 我们知道， `sum` 可以按照某一维度求和， 即把某一维度"压扁"。 那么， 我可以这样压扁前两个维度:

```python
    rgrad = rgrad.sum(tuple([i for i in range(2)]))
```

如果泛化，则是这样:

```python
rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
```

即我们用 `sum` 把多出来的维度压扁。保证形状的统一性。

### Transpose

```python
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # Notice here:
        # if the self.axies is None, then swap the last two dimensions
        # otherwise, swap the dimensions according to the self.axes
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)
```

对于矩阵转置， 首先注意， 我们前向计算的时候只交换最后两个维度。 如 $A : 6 x 6 x 5 x 4$, 则 $A^T : 6 x 6 x 4 x 5$.

另外根据规则

$\frac{\partial L}{\partial A} = (\frac{\partial L}{\partial A^T})^T$

梯度计算就是直接对 `outgrad` 求转置即可。

### Reshape

```python
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return out_grad.reshape(node.inputs[0].shape)
```

根据规则， $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial A.reshape(shape)}.reshape(A.shape)$.

我反向传播的时候只需要把 `outgrad` 的形状重塑回 `A` 的形状就行了。

### broadcasting

```python
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        original_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        # get the original shape and initialize an array 
        # to represent all the dims to be shrinked
        for i, (ori, cur) in enumerate(zip(reversed(original_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
                # if the dimension is the same as an original one, then we should not shrink it
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(original_shape)
        # firstly we sum the dims that are shrinked, then reshape it back to the original shape
```

`broadcast_to` 本身可以广播一个向量， 比如：

```
A:
[[1]
 [2]
 [3]]
A_broadcasted (A broadcasted to (3, 4)):
[[1 1 1 1]
 [2 2 2 2]
 [3 3 3 3]]
```

我们反向传播的时候， 需要做维度压缩。 我们首先创建一个数组， 然后确定哪些维度要压缩。 

```python
for i, (ori, cur) in enumerate(zip(reversed(original_shape), reversed(self.shape))):
    if ori == cur:
        shrink_dims[len(self.shape) - i - 1] = -1
shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
```

这一段代码会把不需要压缩的维度标记为 -1。 之后， 我们过滤掉不需要压缩的维度。

也就是， 经过上述变换后， 我们的广播后的 $A$ 对应数组为 $[-1, 1]$.

需要压缩的为第 1 个维度 (0-indexed).

然后我们用 `sum` 对 `outgrad` 进行压缩。 

假设 

```
outgrad
[[1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]]
```

压缩后即为 

```
outgrad after sum
[[4]
 [4]
 [4]]
```

然后再 `reshape` 到原先的形状。 即为

```
outgrad after sum after reshape
[[4]
 [4]
 [4]]
```

### Summation

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis = self.axes)

    def gradient(self, out_grad, node):
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
```

对于 `summation`， 反向传播需要先 `shape` 再 `broadcast`.

我们可以看个例子：

不妨假设

```
A.shape: (3, 4, 5)
A_summed = array_api.sum(A, axes=(1, ))
A_summed.shape: (3, 5)
```

则 outgrad 形状与 `A_summed` 相同， 为 `(3, 5)`.

反向传播时， 我们先确定 `sum` 的轴为第一个轴， 然后我们 `reshape` :

```
outgrad: (3, 5)
outgrad after reshape: (3, 1, 5)
```

之后我们广播回原来的形状：

```
outgrad: (3, 5)
outgrad after reshape after broadcasting: (3, 4, 5)
```

