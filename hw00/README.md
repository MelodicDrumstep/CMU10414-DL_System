本次 project 是写 `parse_mnist`、 `softmax_Loss`、 `softmax_regression`、`train_softmax`、`train_nn`.

## parse_mnist

就是手写一个 parse mnist 格式图像的函数。

 ```python
 def parse_mnist(image_filename, label_filename):
    with gzip.open(image_filename, 'rb') as img_f, gzip.open(label_filename, 'rb') as lbl_f:
        # Read the magic number and the dimensions of the image data
        magic, num_images = struct.unpack(">II", img_f.read(8))
        # > sets the order to big endian, I is a format character that specifies an unsigned integer.
        rows, cols = struct.unpack(">II", img_f.read(8))

        # Read the label file's magic number and number of labels
        magic, num_labels = struct.unpack(">II", lbl_f.read(8))
        
        # Ensure the number of images and labels are the same
        if num_images != num_labels:
            raise ValueError("The number of images does not match the number of labels.")

        # Read all the images at once (each image is rows * cols bytes)
        images = np.frombuffer(img_f.read(num_images * rows * cols), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols).astype(np.float32)
        images /= 255.0  # Normalize the pixel values to be between 0.0 and 1.0

        # Read all the labels at once
        labels = np.frombuffer(lbl_f.read(num_labels), dtype=np.uint8)

    return images, labels
```

## softmax_loss

套公式，用 `numpy` 实现 `softmax` 的损失函数。 

公式为 $l(z, y) = log(\Sigma_{i=1}^k exp(z_i)) - z_y$.

其中 $z$ 是一个输入样本， $z_i$ 表示模型计算出被分类为第 $i$ 类的概率。

我们可以把所有的输入向量组合成一个矩阵 $Z$, 每一行为一个样本， $Z[x, y]$ 表示第 $x$ 个样本被模型分类为类别 $y$ 的概率。

每一轮的 loss function 就可以写为

$(\Sigma_{z}l(z, y)) / BatchSize =( \Sigma_{z}(log(\Sigma_{i=1}^k exp(z_i)) - z_y)) / BatchSize$

$= (\Sigma_{i}(log(\Sigma_{j=1}^k exp(Z[i, j])) - Z[i, y]))  / BatchSize$

```python
def softmax_loss(Z, y):
    """
    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    return (np.sum(np.log(np.sum(np.exp(Z), axis = 1))) - np.sum(Z[np.arange(y.size), y])) / y.size
```

这里 `np.sum(np.exp(Z), axis = 1)` 表示按行求和， 即先对 $Z$ 矩阵每个元素求 `exp`， 然后按行加起来， 得到一个列向量。 这个列向量再每个元素求 `log`， 再加起来成为一个数。

$Z[np.arange(y.size), y]$ 是一个高级索引， 输出为一个列向量。 即 $np.arange(y. size) = [0, 1,..., y.size - 1]$ 提供行索引， $y$ 提供列索引。 

如果难以理解， 可以看这个例子：

$$
Z = \begin{bmatrix}
a & b &c\\
d & e & f
\end{bmatrix}
$$

$$
y = \begin{bmatrix}
2  \\
0
\end{bmatrix}
$$

则 

$$
Z[np.arange(y.size), y] = \begin{bmatrix}
c  \\
d
\end{bmatrix}
$$

因此， 上述的公式可以正确输出一个 batch 的 softmax loss.

## softmax_regression

机器学习领域有一个词叫做 `epoch`， 表示利用一批次样本计算损失函数梯度并更新参数的一次过程。我们这里要写的这个函数就是完成一次 `softmax regression` 的 `epoch`.

公式是这样的:

$\frac{\partial l(\Theta)}{\partial \Theta} = X^T(Z-I_y) / BatchSize$.

其中 $I_y$ 只有样本点对应的真实分类处元素为 1， 其他元素为 0。

即， $I_y[x, y] = 1$ 若样本 $x$ 的真实类别为 $y$。 否则 $I_y[x, y] = 0$.

总体步骤我都写在注释里了。

```python
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """
    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
    """
    for i in range(0, (y.size + batch - 1) // batch):
        x_batch = X[i * batch : (i + 1) * batch]
        y_batch = y[i * batch : (i + 1) * batch]
        # take the x and y of this pariticular batch
        Z = np.exp(x_batch @ theta)
        # @ is for matrix multiplication
        Z = Z / np.sum(Z, axis = 1, keepdims = True)
        # normalization
        I_y = np.zeros((batch, y.max() + 1))
        I_y[np.arange(batch), y_batch] = 1
        # create I_y
        gradient = (x_batch.T @ (Z - I_y)) / batch
        # calculate the gradient according to the formula
        theta -= lr * gradient
        # renew the paramater theta
```

## nn_regression

这个函数是写一个两层神经网络的 regression epoch。 即 FC -> ReLU -> FC.

第一个 FC： 输入 $x$， 输出 $W_1^Tx$.

ReLU: 输入 $W_1^Tx$, 输出 $ReLU(W_1^Tx)$.

第二个 FC： 输入 $ReLU(W_1^Tx)$, 输出 $W_2^TReLU(W_1^Tx)$.

所以我们的损失函数可以写成

$\Sigma_{i=1}^{BatchSize}l(W_2^TReLU(W_1^Tx_i), y_i)$.

$= l(ReLU(XW_1)W_2, y)$

令

$Z_1 = ReLU(XW_1)$

$G_2 = normalize(exp(Z_1W_2)) - I_y$

$G_1 = 1[Z_1 > 0] (G_2W_2^T)$

我们可以得到

对 $W_2$ 梯度即为 $X^TG_1 / BatchSize$

对 $W_1$ 梯度即为 $Z_1^TG_2 / BatchSize$

最后我用对 $W_1$ 的梯度来更新 $W_1$, 用对 $W_2$ 的梯度来更新 $W_2$.

```python
def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ 
    Run a single epoch of SGD for a two-layer neural network 
    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
    """
    for i in range(0, (y.size + batch - 1) // batch):
        x_batch = X[i * batch : (i + 1) * batch]
        y_batch = y[i * batch : (i + 1) * batch]
        Z1 = x_batch @ W1
        Z1[Z1 < 0] = 0
        # Z1 = ReLU(XW_1)
        G2 = np.exp(Z1 @ W2)
        G2 = G2 / np.sum(G2, axis = 1, keepdims = True)
        I_y = np.zeros((batch, y.max() + 1))
        I_y[np.arange(batch), y_batch] = 1
        # create I_y
        G2 -= I_y
        # G_2 = normalize(exp(Z_1W_2)) - I_y
        G1 = np.zeros_like(Z1)
        G1[Z1 > 0] = 1
        G1 = G1 * (G2 @ W2.T)
        # G_1 = 1[Z_1 > 0] (G_2W_2^T)
        grad1 = x_batch.T @ G1 / batch
        # grad1 = X^T G_1 / batch
        grad2 = Z1.T @ G2 / batch
        # grad2 = Z_1^T G_2 / batch
        W1 -= lr * grad1
        W2 -= lr * grad2
        # renew the parameters W_1 and W_2
```

## train_softmax

我们在实际训练的时候， 希望输出给用户每轮的 loss 值和误差值， 因此， 写一个辅助函数:

```python
def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)
```

训练函数可以写成这样:

```python
def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            # if the cpp version is available, use the cpp version
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))

def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))
```

这里我们试图去调用了 cpp 版本的 `softmax_regression_epoch` 函数。 python 如何调用 cpp? 本次 project 使用的是 `Pybind11` 来实现。

## softmax regression with pybind11

很多时候， 我们用 C / Cpp 改写 python 中的函数， 可以达到更好的性能。 这里我们使用 cpp 改写 `softmax_regression_epoch`， 代码在 `simple_ml_ext.cpp` 中。

我们使用这样的 pybind11 模块来让 python 调用编译后的 cpp 动态库

```c
/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    // we add a function named "softmax_regression_epoch_cpp" to m
    // the following is a lambda expression
    	[](py::array_t<float, py::array::c_style> X,
            // this type corresponding to the numpy array type
            // type of element is float 
            // and use C style memory layout
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            /// this will get the pointer of the numpy array X
            // and cast it to const float *
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            // get the shape of the numpy array X
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
    // these are the argument names of the function
}
```

