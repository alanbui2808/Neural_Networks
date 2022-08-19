from builtins import object
import numpy as np

from cs682.layers import *
from cs682.layer_utils import *


class ConvNet(object):
    """
   A simple convolutional network with the following architecture:

    [conv - bn - relu] x M - adaptive_average_pooling - affine - softmax
    
    "[conv - bn - relu] x M" means the "conv-bn-relu" architecture is repeated for
    M times, where M is implicitly defined by the convolution layers' parameters.
    
    For each convolution layer, we do downsampling of factor 2 by setting the stride
    to be 2. So we can have a large receptive field size.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_sizes=[7],
                 num_classes=10, weight_scale=1e-3, reg=0.0,use_batch_norm=True, 
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer. It is a
          list whose length defines the number of convolution layers
        - filter_sizes: Width/height of filters to use in the convolutional layer. It
          is a list with the same length with num_filters
        - num_classes: Number of output classes
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - use_batch_norm: A boolean variable indicating whether to use batch normalization
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = []
        self.conv_params = []
        self.normalization = use_batch_norm

        ############################################################################
        # TODO: Initialize weights and biases for the simple convolutional         #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params.                                                 #
        #                                                                          #
        # IMPORTANT:                                                               #
        # 1. For this assignment, you can assume that the padding                  #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. You need to         #
        # carefully set the `pad` parameter for the convolution.                   #
        #                                                                          #
        # 2. For each convolution layer, we use stride of 2 to do downsampling.    #
        ############################################################################
        
        C, H, W = input_dim
        self.num_conv_layers = len(num_filters)
        C_input = None

        # Convolutional Layer:
        for i in range(self.num_conv_layers):
          C_input = C if C_input is None else num_filters[i - 1]
          F, FH, FW = num_filters[i], filter_sizes[i], filter_sizes[i]

          # ConvLayer:
          stride = 1 if i == 0 else 2
          pad = (FH - 1)//2 # rounding is OK
          self.conv_params.append({'stride': stride, 'pad': pad})
          
          self.params['W' + str(i + 1)] = weight_scale * np.random.randn(F, C_input, FH, FW)
          self.params['b' + str(i + 1)] = np.zeros(F)
          # After Conv: (N, F, H', W')

          # BN:
          if self.normalization:
            self.params['gamma' + str(i + 1)] = np.ones(F) # (F, )
            self.params['beta' + str(i + 1)] = np.zeros(F) # (F, )
            # out = (N, F, H', W')

            # BN_params:
            self.bn_params.append({'mode': 'train'})
        
        # Affine Layer:
        self.params['W' + str(self.num_conv_layers + 1)] = weight_scale * np.random.randn(num_filters[-1], num_classes)
        self.params['b' + str(self.num_conv_layers + 1)] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        scores = None
        mode = 'test' if y is None else 'train'
        if self.normalization:
          for bn_param in self.bn_params:
            bn_param['mode'] = mode
        # print('Y is none {}'.format(y is None))
        ############################################################################
        # TODO: Implement the forward pass for the simple convolutional net,       #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        
        caches = []
        X_input = X
        X_output = None

        # Conv Layer Forward: 
        for i in range(self.num_conv_layers):
          conv_attr = bn_attr = None

          W, b, conv_param = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)], self.conv_params[i]
          conv_attr = (W, b, conv_param)

          if self.normalization:
            gamma, beta, bn_param = self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)], self.bn_params[i]
            bn_attr = (gamma, beta, bn_param)
          
          # Conv-BN-ReLU: (N, F, H, W) --> (N, F, H', W')
          X_output, cache = conv_bn_relu_forward(X_input, conv_attr, bn_attr, self.normalization)
          caches.append(cache)
          # print(cache[1])
          # print('normalize {}'.format(self.normalization))

          # Update for next layer
          X_input = X_output
        
        # Average Adaptive Pooling: (N, C, H, W) --> (N, C)
        X_AAP, cache = adaptive_avg_pool_forward(X_output)
        caches.append(cache)
        
        # Affine: (N, C) --> (N, # of classes)
        W, b = self.params['W' + str(self.num_conv_layers + 1)], self.params['b' + str(self.num_conv_layers + 1)]
        X_affine, cache = affine_forward(X_AAP, W, b)
        caches.append(cache)

        scores = X_affine

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
          return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the simple convolutional net,      #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Loss:
        loss, dL_dXaffine = softmax_loss(X_affine, y)

        for i in range(self.num_conv_layers + 1): # M ConvLayer + 1 Affine Layer
          W = self.params['W' + str(i + 1)]
          loss += 0.5 * self.reg * np.sum(W * W)

        # Affine:
        dX_APP, dW_affine, db_affine = affine_backward(dL_dXaffine, caches[-1])
        grads['W' + str(self.num_conv_layers + 1)] = dW_affine + (self.reg * self.params['W' + str(self.num_conv_layers + 1)])
        grads['b' + str(self.num_conv_layers + 1)] = db_affine
        # print('Finish backward pass for Affine')

        # Average Adaptive Pooling:
        dX_conv = adaptive_avg_pool_backward(dX_APP, caches[-2])
        # print('Finish backward pass for AAP')

        dXout = dX_conv
        # Conv Layer Backward:
        for i in reversed(range(self.num_conv_layers)):
          cache = caches[i]
          dX, dW, db, dgamma, dbeta = conv_bn_relu_backward(dXout, cache, self.normalization)

          grads['W' + str(i + 1)] = dW + (self.reg * self.params['W' + str(i + 1)])
          grads['b' + str(i + 1)] = db
          if self.normalization:
            grads['gamma' + str(i + 1)] = dgamma
            grads['beta' + str(i + 1)] = dbeta
          
          # Update for next backward pass
          dXout = dX

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

##########################################################################################
#                                   HELPER METHODS                                       #
##########################################################################################

def conv_bn_relu_forward(X, conv_attr, bn_attr, normalization):
  conv_cache = bn_cache = relu_cache = None
  
  # Conv Layer: (N, C, H, W) --> (N, F, H', W')
  W, b, conv_param = conv_attr
  conv, conv_cache = conv_forward_naive(X, W, b, conv_param)

  # Batch Norm: (N, F, H', W') --> (N, F, H', W')
  bn = conv

  if normalization:
    gamma, beta, bn_param = bn_attr
    bn, bn_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
  
  # ReLU: (N, F, H', W') --> (N, F, H', W')
  relu, relu_cache = relu_forward(bn)

  out, cache = relu, (conv_cache, bn_cache, relu_cache)

  return out, cache

def conv_bn_relu_backward(dout, cache, normalization):
  dX = dW = db = dgamma = dbeta = None
  conv_cache, bn_cache, relu_cache = cache

  # Relu:
  drelu = relu_backward(dout, relu_cache)

  # BN:
  dbn = drelu
  if normalization:
    dbn, dgamma, dbeta = spatial_batchnorm_backward(drelu, bn_cache)
  
  # Conv Layer:
  dX, dW, db = conv_backward_naive(dbn, conv_cache)

  return dX, dW, db, dgamma, dbeta

  








