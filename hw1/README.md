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

Tensor 是计算图中结点的类型。 我们还需要定义运算的类型， 接下来看 `Op` 类。

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

class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(0, a)

    def gradient(self, out_grad, node):
        data = node.realize_cached_data().copy()
        data[data > 0] = 1
        return out_grad * Tensor(data)

class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        # \partial L / \partial A = \partial L / \partial C * B^T
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
```

我们自定义了 `compute` 函数， 用于正向计算， 和 `gradient` 函数， 计算梯度， 用于反向传播。 

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