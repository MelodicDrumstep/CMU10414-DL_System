"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
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
    ### END YOUR CODE

def softmax_loss(Z, I_y):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    return (ndl.log(ndl.exp(Z).sum((1,))).sum() - (I_y * Z).sum()) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

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

    ### BEGIN YOUR SOLUTION
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
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    I_y = np.zeros((y.shape[0], h.shape[-1]))
    I_y[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(I_y)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
