### Activation Function Implementations:

Implementation of `activations.Linear`:

```python
class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        return dY

```

Implementation of `activations.Sigmoid`:

```python
class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return ...

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        return ...

```

Implementation of `activations.ReLU`:

```python
class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return np.maximum(0, Z)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        
        return np.where(Z < 0, 0, dY)

```

Implementation of `activations.SoftMax`:

```python
class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        m = np.max(Z, axis=1, keepdims=True)
        expShifted = np.exp(Z - m)
        softmaxOutput = expShifted/np.sum(expShifted, axis=1, keepdims=True)

        return softmaxOutput

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        # print("Z: ", Z.shape)
        # print("dy", dY.shape)
        
        dLdZ = np.empty(Z.shape)
        softmaxZ = self.forward(Z)

        for idx in range(0,softmaxZ.shape[0]):
            # print(softmaxZ[idx])
            # print(softmaxZ[idx, :][:, None])
            currPoint = softmaxZ[idx, :][:, None]
            currdY = dY[idx,:][:, None]
            jacobian = -currPoint @ currPoint.T
            np.fill_diagonal(jacobian, np.array([softmax * (1 - softmax) for softmax in currPoint]))

            dLdZ[idx, :][:, None] = jacobian @ currdY
        
        return dLdZ

```


### Layer Implementations:

Implementation of `layers.FullyConnected`:

```python
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

```

Implementation of `layers.Pool2D`:

```python
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
        pad = self.pad
        kernel_height, kernel_width = self.kernel_shape

        ### BEGIN YOUR CODE ###
        #Padding
        if self.pad != (0, 0):
            xPad = np.pad(X, ((0,), (pad[0],), (pad[1],), (0,)), constant_values=(0))          
        else:
            xPad = X

        hOut = int(1 + (in_rows + 2*pad[0] - kernel_height) / self.stride)
        wOut = int(1 + (in_cols + 2*pad[1] - kernel_width) / self.stride)

        #Declaring preactivation output
        X_pool = np.zeros((n_examples, hOut, wOut, channels))
        indices = np.zeros((hOut, wOut, channels, n_examples), dtype=int)

        # implement the forward pass
        for hIdx in range(hOut):
            for wIdx in range(wOut):
                hStart = hIdx * self.stride
                hEnd = hIdx * self.stride + kernel_height
                wStart = wIdx * self.stride
                wEnd = wIdx * self.stride + kernel_width

                xWindow = xPad[:, hStart:hEnd, wStart:wEnd, :]

                if self.mode == 'max':
                    X_pool[:, hIdx, wIdx, :] = self.pool_fn(xWindow, axis = (1, 2))
                    
                    for c in range(channels):
                        flat = xWindow.flatten()
                        argmax = np.argmax(np.split(flat, n_examples), axis = 1)
                        indices[hIdx, wIdx, c] = argmax

                elif self.mode == 'average':
                    X_pool[:, hIdx, wIdx, :] = self.pool_fn(xWindow, axis=(1, 2))

        # cache any values required for backprop

        if self.mode == 'max':
            self.cache['indices'] = indices
        self.cache['X'] = X

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
        # print("indices: ", indices[0,0])
        X = self.cache['X']
        batch_size, out_rows, out_cols, channels = dLdY.shape
        dX = np.zeros_like(X)
        hOut = dLdY.shape[1]
        wOut = dLdY.shape[2]
        kernel_height, kernel_width = self.kernel_shape
        pad = self.pad

        if self.pad != (0, 0):
            xPad = np.pad(X, ((0,), (pad[0],), (pad[1],), (0,)), constant_values=(0))          
        else:
            xPad = X

        for hIdx in range(hOut):
            for wIdx in range(wOut):
                for c in range(channels):
                    hStart = hIdx * self.stride
                    hEnd = hIdx * self.stride + kernel_height
                    wStart = wIdx * self.stride
                    wEnd = wIdx * self.stride + kernel_width
                    xWindow = xPad[:, hStart:hEnd, wStart:wEnd, c]
                if self.mode == "max":
                    maxIdx = np.unravel_index(indices[hIdx, wIdx, c, :], (kernel_height, kernel_width))
                    
                    dX[:, hStart:hEnd, wStart:wEnd, c] += dLdY[:, hIdx, wIdx, c]

                elif self.mode == "average":
                    dy = dLdY[:, hIdx, wIdx, c]
                    dX[:, hStart:hEnd, wStart:wEnd, c] += self.averageMatrix(dy)
                    
        ### END YOUR CODE ###
        return dX
    
    def averageMatrix(self, dLdY):
        kernel_height, kernel_width = self.kernel_shape
        average = dLdY / (kernel_height * kernel_width)
        return np.ones(self.kernel_shape) * average

```

Implementation of `layers.Conv2D.__init__`:

```python
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

```

Implementation of `layers.Conv2D._init_parameters`:

```python
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

```

Implementation of `layers.Conv2D.forward`:

```python
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

```

Implementation of `layers.Conv2D.backward`:

```python
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

```


### Loss Function Implementations:

Implementation of `losses.CrossEntropy`:

```python
class CrossEntropy(Loss):
    """Cross entropy loss function."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """
        ### YOUR CODE HERE ###
        m = Y.shape[0]
        loss = np.sum(np.diag(Y @ np.log(Y_hat).T))
        return (-1/m) * loss

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass of cross-entropy loss.
        NOTE: This is correct ONLY when the loss function is SoftMax.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the gradient of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        ### YOUR CODE HERE ###
        m = Y.shape[0]
        return (-1/m) * (np.divide(Y,Y_hat))

```


### Model Implementations:

Implementation of `models.NeuralNetwork.forward`:

```python
    def forward(self, X: np.ndarray) -> np.ndarray:
        """One forward pass through all the layers of the neural network.

        Parameters
        ----------
        X  design matrix whose must match the input shape required by the
           first layer

        Returns
        -------
        forward pass output, matches the shape of the output of the last layer
        """
        ### YOUR CODE HERE ###
        # Iterate through the network's layers.
        output = X

        for layer in self.layers:
            output = layer.forward(output)
        return output

```

Implementation of `models.NeuralNetwork.backward`:

```python
    def backward(self, target: np.ndarray, out: np.ndarray) -> float:
        """One backward pass through all the layers of the neural network.
        During this phase we calculate the gradients of the loss with respect to
        each of the parameters of the entire neural network. Most of the heavy
        lifting is done by the `backward` methods of the layers, so this method
        should be relatively simple. Also make sure to compute the loss in this
        method and NOT in `self.forward`.

        Note: Both input arrays have the same shape.

        Parameters
        ----------
        target  the targets we are trying to fit to (e.g., training labels)
        out     the predictions of the model on training data

        Returns
        -------
        the loss of the model given the training inputs and targets
        """
        ### YOUR CODE HERE ###
        # Compute the loss.
        # Backpropagate through the network's layers.
        loss = self.loss(target, out)

        dLdOut = self.loss.backward(target, out)
        for layer in self.layers[::-1]: #Layers in reverse
            dLdOut = layer.backward(dLdOut)

        return loss

```

Implementation of `models.NeuralNetwork.predict`:

```python
    def predict(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make a forward and backward pass to calculate the predictions and
        loss of the neural network on the given data.

        Parameters
        ----------
        X  input features
        Y  targets (same length as `X`)

        Returns
        -------
        a tuple of the prediction and loss
        """
        ### YOUR CODE HERE ###
        # Do a forward pass. Maybe use a function you already wrote?
        # Get the loss. Remember that the `backward` function returns the loss.
        Yh = self.forward(X)
        L = self.backward(Y, Yh)

        return Yh, L

```

