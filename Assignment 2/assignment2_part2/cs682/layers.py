from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # Convert any (N, (d1, ...,dk)) --> (N, (d1*...*dk)) shape
    X = x.reshape(x.shape[0], -1)
    
    # Affine function
    out = np.dot(X, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    df_dY = dout
    X = x.reshape(x.shape[0], -1)
    dY_dX, dY_dW = w.T, X.T
    
    dx = np.dot(df_dY, dY_dX).reshape(x.shape) # reshape so it aligns with original x
    dw = np.dot(dY_dW, df_dY)
    db = np.sum(df_dY, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dout[x <= 0] = 0 
    dx = dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    is_layernorm = bn_param.get('layernorm', 0)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    # print(mode)
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################

        mu, var = np.mean(x, axis=0), np.var(x, axis=0)
        std = np.sqrt(var + eps) # avoid 0 std during normalizing
        # Normalize
        x_nor = (x - mu) / std
        # Scale: y = gamma * x_nor + beta
        out = gamma*x_nor + beta
        
        # Update:
        if not is_layernorm:
            running_mean = momentum*running_mean + (1-momentum)*mu
            running_var = momentum*running_var + (1-momentum)*(std**2)
        
        cache = {'x': x, 'mu': mu, 'var': var, 'std': std, 'x_nor': x_nor, 'gamma': gamma, 'beta': beta, 'axis': is_layernorm}
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        std = np.sqrt(running_var) + eps
        x_nor = (x - running_mean) / std
        out = gamma*x_nor + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    x, std, x_nor, gamma, axis = cache['x'], cache['std'], cache['x_nor'], cache['gamma'], cache['axis']
    N, D = x.shape
    
    dx_nor = dout * gamma 
    
    dx = (1/std * dx_nor) + (-1/(N*std) * np.sum(dx_nor, axis=0)) + (-1/(N*std) * x_nor * np.sum(dx_nor * x_nor, axis=0))
    
    dbeta = np.sum(dout, axis=axis)
    dgamma = np.sum(dout * x_nor, axis=axis)

    assert dx.shape == (N,D)
    assert dgamma.shape == (D,)
    assert dbeta.shape == (D,)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
    ln_param['mode'], ln_param['layernorm'] = 'train', 1
    
    out, cache = batchnorm_forward(x.T, gamma.reshape(-1, 1), beta.reshape(-1, 1), ln_param)
    out = out.T
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)
    dx = dx.T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See https://compsci682-fa18.github.io/notes/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p # filter out weights with p
        
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = dout * mask # similar idea to ReLU

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    # Padding:
    # https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
    padding = ((0,0),(0,0),(pad,pad),(pad,pad)) 
    x_padding = np.pad(x, padding, 'constant')
    N, C, H_padding, W_padding = x_padding.shape

    # ConV layer:
    # assert (W - FW + 2*pad) % stride == 0, 'Cannot convolve filter over image width'
    # assert (H - FW + 2*pad) % stride == 0, 'Cannot convolve filter over image height'
    # rounding is ok
    Wout = ((W - FW + 2*pad)//stride) + 1
    Hout = ((H - FH + 2*pad)//stride) + 1

    out = np.zeros((N, F, Hout, Wout))

    for n in range(N):
      for f in range(F):
        for i in range(0, (H_padding-FH+1), stride):
          for j in range(0, (W_padding-FW+1), stride):
            kernel, bias = w[f], b[f]
            image_portion = x_padding[n, :, i:i+FH, j:j+FW]

            out[n,f, i//stride, j//stride] = np.sum(image_portion * kernel) + bias

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Padding:
    padding = ((0,0),(0,0),(pad,pad),(pad,pad)) 
    x_padding = np.pad(x, padding, 'constant')
    N, C, H_padding, W_padding = x_padding.shape

    # Backpropogation:
    dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
    dx_padding = np.zeros(x_padding.shape)

    for n in range(N):
      for f in range(F):
        for i in range(0, (H_padding - FH + 1), stride):
          for j in range(0, (W_padding - FW + 1), stride):
            kernel = w[f]
            dy = dout[n, f, i//stride, j//stride]
            image_portion = x_padding[n, :, i:i+FH, j:j+FW]

            dw[f] += dy * image_portion
            db[f] += dy
            dx_padding[n, :, i:i+FH, j:j+FW] += dy * kernel

    # Take only pixels from original X.
    dx = dx_padding if pad == 0 else dx_padding[:,:, pad:-pad, pad:-pad]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    
    N, C, H, W = x.shape
    PH, PW = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    # assert (H - PH) % stride == 0, 'Cannot convolve over image width given stride'
    # assert (W - PW) % stride == 0, 'Cannot convolve over image height given stride'
    # rounding is ok
    Hout = (H - PH)//stride + 1
    Wout = (W - PW)//stride + 1

    # Max Pooling
    out = np.zeros((N, C, Hout, Wout))

    for n in range(N):
      for c in range(C):
        for i in range(0, (H - PH + 1), stride):
          for j in range(0, (W - PW + 1), stride):
            image_portion = x[n, c, i:i+PH, j:j+PW]

            out[n, c, i//stride, j//stride] = np.max(image_portion)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    
    x, pool_param = cache
    N, C, H, W = x.shape
    PH, PW = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    # Backpropogation:
    dx = np.zeros(x.shape)

    for n in range(N):
      for c in range(C):
        for i in range(0, (H - PH + 1), stride):
          for j in range(0, (W - PW + 1), stride):
            dz = dout[n, c, i//stride, j//stride]
            image_portion = x[n, c, i:i+PH, j:j+PW]
            max_value = np.max(image_portion)
            max_indices = np.argwhere(image_portion == max_value)

            for ii,jj in max_indices:
              # ii and jj are indices w.r.t to image_portion, not original image
              dx[n, c, i+ii, j+jj] += dz
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def adaptive_avg_pool_forward(x):
    """
    Computes the forward pass of the adaptive average pooling layer
    
    Input:
    - x: Input data of shape (N, C, H, W)
    
    Returns of a tuple of:
    - out: Output data, of shape (N, C)
    - cache: (x,)
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement the forward pass for adaptive average pooling.          #
    ###########################################################################
    N, C, H, W = x.shape
    
    # Average pooling:
    out = np.zeros((N,C))

    for n in range(N):
      for c in range(C):
        out[n, c] = np.mean(x[n, c, :, :])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x)

    return out, cache

def adaptive_avg_pool_backward(dout, cache):
    """
    Computes the forward pass of the adaptive average pooling layer
    
    Inputs:
    - dout: Upstream derivatives.
    - cache: x as in global_avg_pool_backward
    
    Returns:
    - dx: gradient with respect x
    
    """
    dx = None
    ###########################################################################
    # TODO: Implement the backward pass for adaptive average pooling.          #
    ###########################################################################

    x = cache
    N, C, H, W = x.shape
    
    # Backpropogation:
    dx = np.zeros(x.shape)

    for n in range(N):
      for c in range(C):
        dx[n, c, :, :] += 1/(H*W) * dout[n, c]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx    


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = x.shape

    # Reshape into X into (N*H*W,C) shape:
    x_reshape = x.transpose(0,2,3,1).reshape(N*H*W, C)
    # BN:
    out, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param)
    # Reshape back into (N,C,H,W) shape
    out = out.reshape(N, H, W, C).transpose(0,3,1,2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = dout.shape
    dout_reshape = dout.transpose(0,2,3,1).reshape(N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_reshape, cache)
    dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)

    assert(dx.shape == (N,C,H,W))
    assert(dgamma.shape == (C,))
    assert(dbeta.shape == (C,))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    
    N, C, H, W = x.shape

    # Reshape x so it fits the implementation of layernorm (batchnorm)
    groups = (N*G, (C//G)*H*W)
    x_gn = np.reshape(x, groups)
    # Similar idea of transpose x in layernorm and reuse batchnorm implementation (unnecessary)
    # This is for standard implementation derived from BN
    x_gn = x_gn.T

    # Mean and std:
    mu = np.mean(x_gn, axis=0)
    var = np.var(x_gn, axis=0)
    std = np.sqrt(var + eps)
    # Normalizie:
    x_nor = (x_gn - mu) / std

    # Reshape back for output:
    x_nor = x_nor.T
    x_nor = np.reshape(x_nor, (N, C, H, W))

    # output:
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)
    out = gamma*x_nor + beta

    cache = {'mu': mu, 'std': std, 'x_nor': x_nor, 'gamma': gamma, 'G': G}
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    N, C, H, W = dout.shape
    mu, std, x_nor, gamma, G = cache['mu'], cache['std'], cache['x_nor'], cache['gamma'], cache['G']

    dx_nor = dout * gamma
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True) # keep the shape to be (1,C,1,1)
    dgamma = np.sum(dout*x_nor, axis=(0,2,3), keepdims=True) # keep the shape to be (1,C,1,1)

    dx_nor = np.reshape(dx_nor, (N*G, C//G*H*W)).T # reshape for LN
    x_nor = np.reshape(x_nor, (N*G, C//G*H*W)).T # reshape for LN

    N_g, D_g = x_nor.shape

    # Similar implementation to BN:
    dx = (1/std * dx_nor) + (-1/(N_g*std) * np.sum(dx_nor, axis=0)) + (-1/(N_g*std) * x_nor * np.sum(dx_nor * x_nor, axis=0))
    dx = np.reshape(dx.T, (N, C, H, W))

    assert(dx.shape == (N,C,H,W))
    assert(dbeta.shape == (1,C,1,1)) # equiv to (C,)
    assert(dgamma.shape == (1,C,1,1)) # equiv to (C,)
    ###########################################################################
    #                         \     END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
