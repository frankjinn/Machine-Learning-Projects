"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

from matplotlib.bezier import inside_circle
import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache: OrderedDict = {"out":0, "X":0}  # cache for backprop "W":0, "X":0, "b":0, 
        self.gradients: OrderedDict = {"W":np.zeros_like(W), "b":np.zeros_like(b)}  # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        
        # perform an affine transformation and activation
        Z = X@self.parameters['W'] + self.parameters['b']
        out = self.activation(Z)
        
        # store information necessary for backprop in `self.cache`
        self.cache["Z"] = Z
        self.cache["X"] = X
        ### END YOUR CODE ###

        return out

    def backward(self, dLdOut: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdOut  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        Z = self.cache["Z"] #output of layer
        X = self.cache["X"] #input of layer
        W = self.parameters["W"]
        dLdZ = self.activation.backward(Z, dLdOut) #Loss with respect to param before activation

        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        dLdW = X.T @ (dLdZ)
        dLdb = dLdZ.sum(axis = 0, keepdims = True)
        dLdX = dLdZ @ W.T

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.

        self.gradients["W"] = dLdW
        self.gradients["b"] = dLdb

        ### END YOUR CODE ###

        return dLdX #Originally dX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)
        pad = self.pad

        ### BEGIN YOUR CODE ###
        #Padding
        if self.pad != (0, 0):
            xPad = np.pad(X, ((0,), (pad[0],), (pad[1],), (0,)), constant_values=(0))          
        else:
            xPad = X

        hOut = int(1 + (in_rows + 2*pad[0] - kernel_height) / self.stride)
        wOut = int(1 + (in_cols + 2*pad[1] - kernel_width) / self.stride)

        #Declaring preactivation output
        Z = np.empty((n_examples, hOut, wOut, out_channels))

        # implement a convolutional forward pass
        for hIdx in range(hOut):
            for wIdx in range(wOut):
                window = xPad[:, hIdx*self.stride : hIdx*self.stride + kernel_height,
                              wIdx*self.stride : wIdx*self.stride + kernel_width, :]
                for c in range(out_channels):
                    Z[:, hIdx, wIdx, c] = np.sum(window*W[:, :, :, c], axis=(1,2,3)) + b[0, c]
                # Z[:, wIdx, hIdx, :] = np.einsum('...whc, hwcd -> ...d', window, W) + b[0]

        # cache any values required for backprop
        self.cache["X"] = X
        self.cache["Z"] = Z

        #Apply activation
        out = self.activation(Z)

        # assert out.shape == (n_examples, wOut, hOut, out_channels), "Wrong output shape"
        ### END YOUR CODE ###

        return out
    
    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###
        X = self.cache["X"]
        Z = self.cache["Z"]

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)
        pad = self.pad

        ### BEGIN YOUR CODE ###
        #Padding
        if self.pad != (0, 0):
            xPad = np.pad(X, ((0,), (pad[0],), (pad[1],), (0,)), constant_values=(0))          
        else:
            xPad = X

        hOut = dLdY.shape[1]
        wOut = dLdY.shape[2]
        padH, padW = self.pad

        #padding X
        if self.pad != (0, 0):
            xPad = np.pad(X, ((0,), (pad[0],), (pad[1],), (0,)), constant_values=(0))          
        else:
            xPad = X
        
        #Gradients
        paddedDx = np.zeros(xPad.shape)
        dW = np.zeros(W.shape)
        dLdZ = self.activation.backward(Z, dLdY)

        # perform a backward pass
        for hIdx in range(hOut):
            for wIdx in range(wOut):
                hStart = hIdx * self.stride
                hEnd = hIdx * self.stride + kernel_height
                wStart = wIdx * self.stride
                wEnd = wIdx * self.stride + kernel_width

                xWindow = xPad[:, hStart:hEnd, wStart:wEnd, :]
                paddedDx[:, hStart:hEnd, wStart:wEnd, :] += np.einsum('hwio, bo -> bhwi', W, dLdZ[:, hIdx, wIdx, :])
                
                for c in range(out_channels):
                    dW[:,:,:,c] += (xWindow * dLdY[:,hIdx,wIdx,c][:,None,None,None]).sum(axis=0)
                   

                
                
                # paddedDx[:, hIdx * self.stride : hIdx * self.stride + kernel_height,
                #         wIdx * self.stride : wIdx * self.stride + kernel_width, :] += \
                # 1 
                        
        dX = paddedDx[:,pad[0]:-pad[0], pad[1]:-pad[1], :]
        db = dLdY.sum(axis=(0, 1, 2)).reshape(1, -1)
        
        self.gradients["W"] = dW
        self.gradients["b"] = db

        ### END YOUR CODE ###

        return dX

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        n_examples, in_rows, in_cols, channels = X.shape
        pad = self.pad[0]

        ### BEGIN YOUR CODE ###
        #Padding
        if self.pad != (0, 0):
            xPad = np.pad(X, ((0,), (pad,), (pad,), (0,)), constant_values=(0))          
        else:
            xPad = X

        hOut = int((in_cols + 2*pad - self.kernel_shape[0]) / self.stride) + 1
        wOut = int((in_rows + 2*pad - self.kernel_shape[1]) / self.stride) + 1

        #Declaring preactivation output
        X_pool = np.zeros((n_examples, wOut, hOut, channels))
        indices = np.zeros_like(X_pool, dtype=int)

        # implement the forward pass
        for rowIdx in range(hOut):
            for colIdx in range(wOut):
                rowStart = rowIdx * self.stride
                rowEnd = rowStart + self.kernel_shape[1]
                colStart = colIdx * self.stride
                colEnd = colStart + self.kernel_shape[0]

                window = xPad[:, rowStart:rowEnd, colStart:colEnd, :]

                if self.mode == 'max':
                    X_pool[:, rowIdx, colIdx, :] = np.max(window, axis = (1, 2))
                    print(np.argmax(window, axis=1, keepdims=True).shape)
                    indices[:, rowIdx, colIdx, :] = np.argmax(window, axis=1, keepdims=True)

                elif self.mode == 'average':
                    X_pool[:, rowIdx, colIdx, :] = self.pool_fn(window, axis=(1, 2))

        # cache any values required for backprop

        if self.mode == 'max':
            self.cache['indices'] = indices

        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        indices = self.cache['indices']

        # perform a backward pass
        if self.mode == 'max':
            dX = self.backwardMaxPool(dLdY, indices)
        elif self.mode == 'average':
            dX = self.backward_average_pooling(dLdY)
        
        ### END YOUR CODE ###
        return dX
    
    def backwardMaxPool(self, dLdY, indices):
        """Backward pass for max pooling"""
        batch_size, out_rows, out_cols, channels = dLdY.shape
        dX = np.zeros_like(indices)

        for rowIdx in range(out_rows):
            for colIdx in range(out_cols):
                    rowStart = rowIdx * self.stride
                    rowEnd = rowStart + self.kernel_shape[1]
                    colStart = colIdx * self.stride
                    colEnd = colStart + self.kernel_shape[0]
                    dX[:, rowStart:rowEnd,
                        colStart:colEnd, :] = dLdY[:, rowIdx, colIdx, :] / (self.kernel_shape[0] * self.kernel_shape[1])
        return dX

    def backwardAveragePooling(self, dLdY, indices):
        """Backward pass for max pooling"""
        batch_size, out_rows, out_cols, channels = dLdY.shape
        dX = np.zeros_like(indices)

        for rowIdx in range(out_rows):
            for colIdx in range(out_cols):
                for c in range(channels):
                    idx = indices[:, rowIdx, colIdx, c]
                    dX[:, rowIdx * self.stride + idx // self.kernel_shape[0],
                        colIdx * self.stride + idx % self.kernel_shape[1], c] = dLdY[:, rowIdx, colIdx, c]
        return dX
    
class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
