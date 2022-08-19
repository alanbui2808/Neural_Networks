from builtins import range
from builtins import object
import numpy as np

from cs682.layers import *
from cs682.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        layer_dims = [input_dim] + hidden_dims + [num_classes]
        
        for i in range(self.num_layers): # L + 1
            layer_id = i + 1
            dim_input, dim_output = layer_dims[i], layer_dims[i + 1]
            
            # E.g: [D, H1, H2, C] --> (D,H1) --> (H1, H2) --> (H2, C)
            self.params['W' + str(layer_id)] = weight_scale * np.random.randn(dim_input, dim_output)
            self.params['b' + str(layer_id)] = np.zeros(dim_output)
            
            if self.normalization and layer_id < self.num_layers:
                self.params['gamma' + str(layer_id)] = np.ones(dim_output) 
                self.params['beta' + str(layer_id)] = np.zeros(dim_output)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        # X(i) = W(i)X(i - 1) + b(i) of layer i
        output = [None for i in range(self.num_layers + 1)] # Xi = result(FCi)
        output[0] = X
        # cache[i] = {X(i - 1), W(i), b(i)} of layer i
        caches = [None for i in range(self.num_layers)] # (Xi-1, W_i, b_i)
        
        # Forward Pass:
        for i in range(self.num_layers):
            layer_id = i + 1
            X = W = b = gamma = beta = bn_param = None
            
            X, W, b = output[i], self.params['W' + str(layer_id)], self.params['b' + str(layer_id)]
            if self.normalization and layer_id < self.num_layers:
                gamma = self.params['gamma' + str(layer_id)]
                beta = self.params['beta' + str(layer_id)]
                bn_params = self.bn_params[i]
            
            computer_layer, layer_params = None, None
            
            if layer_id == self.num_layers: # last layer is simply Affine
                compute_layer = affine_forward
                layer_params = (X, W, b)
            else:        
                compute_layer = affine_normalize_relu_dropout_forward
                layer_params = (X, W, b, gamma, beta, bn_param, self.normalization, self.use_dropout, self.dropout_param)
            
            FC, FC_cache = compute_layer(*layer_params)
            output[i + 1] = FC # record next Xi for next (Fully Connected) layer
            caches[i] = FC_cache # record the cache of this layer
        
        # Scores:
        scores = output[-1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # Compute the loss:
        loss, dL_dsoftmax = softmax_loss(scores, y)
        for layer_id in range(1, self.num_layers + 1):
            W = self.params['W' + str(layer_id)]
            loss += 0.5 * self.reg * np.sum(W * W)
        
        # Compute the gradient:
        # upstream derivatives
        douts = [None for i in range(self.num_layers)] 
        douts[-1] = dL_dsoftmax 
        
        for i in reversed(range(self.num_layers)):
            layer_id = i + 1
            dout, cache = douts[i], caches[i]
            dX = dW = db = dgamma = dbeta = None
            
            if layer_id == self.num_layers: # last layer is simply Affine 
                dX, dW, db = affine_backward(dout, cache)
            else:
                compute_gradient = affine_normalize_relu_dropout_backward
                gradient_params = (dout, cache, self.normalization, self.use_dropout)
                
                dX, dW, db, dgamma, dbeta = compute_gradient(*gradient_params) # gradient given cache and upstream derivative
            
            grads['W' + str(layer_id)] = dW + (self.reg * self.params['W' + str(layer_id)])
            grads['b' + str(layer_id)] = db
            
            if self.normalization and layer_id < self.num_layers:
                grads['gamma' + str(layer_id)] = dgamma
                grads['beta' + str(layer_id)] = dbeta
            
            if i > 0:
                douts[i - 1] = dX # update upstream derivative for next iteration

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
#-------------------------------------HELPER FUNCTIONS--------------------------------
def affine_normalize_relu_dropout_forward(x, w, b, gamma, beta, bn_param, normalization, use_dropout, dropout_param):
    fc_cache = nor_cache = relu_cache = dropout_cache = None

    # Affine:
    a, fc_cache = affine_forward(x, w, b)
    a_nor = a

    # Normalization [OPTIONAL]
    if normalization == 'batchnorm':
      a_nor, nor_cache = batchnorm_forward(a, gamma, beta, bn_param)
    elif normalization == 'layernorm':
      a_nor, nor_cache = layernorm_forward(a, gamma, beta, bn_param)
    
    # ReLU:
    c, relu_cache = relu_forward(a_nor)
    out = c

    # Dropout [OPTIONAL]
    if use_dropout:
      out, dropout_cache = dropout_forward(c, dropout_param)
    
    cache = (fc_cache, nor_cache, relu_cache, dropout_cache)
    return out, cache

def affine_normalize_relu_dropout_backward(dout, cache, normalization, use_dropout):
    fc_cache, nor_cache, relu_cache, dropout_cache = cache
    dx = dw = db = dgamma = dbeta = None

    ddropout = dout
    # Dropout [OPTIONAL]
    if use_dropout:
      ddropout = dropout_backward(dout, dropout_cache)
    
    # ReLU
    drelu = relu_backward(ddropout, relu_cache)
    dnor = drelu

    # Normalization [OPTIONAL]
    if normalization == 'batchnorm':
      dnor, dgamma, dbeta = batchnorm_backward_alt(drelu, nor_cache)
    elif normalization == 'layernorm':
      dnor, dgamma, dbeta = layernorm_backward(drelu, nor_cache)
    
    # Affine:
    dx, dw, db = affine_backward(dnor, fc_cache)

    return dx, dw, db, dgamma, dbeta
