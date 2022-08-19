import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, num_classes = np.shape(X)[0], np.shape(W)[1]
    
    for i in range(num_train):
        scores = np.dot(X[i], W)
        new_scores = scores - np.max(scores) # shift by max(scores) to avoid numerical instability
        # Calculate software this way to support calculating gradient
        softmax = np.exp(new_scores) / np.sum(np.exp(new_scores))
        
        # softmax[y[i]] = prob of true label
        loss += -np.log(softmax[y[i]])
        
        x_i = X[i]
        for j in range(num_classes):
            dW[:, j] += x_i * softmax[j]
            if j == y[i]:
                dW[:, y[i]] -= x_i
    
    # Average
    loss /= num_train
    dW /= num_train
    
    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, C = np.shape(X)[0], np.shape(W)[1]
    
    scores = np.dot(X, W)
    
    # Avoid numerical instability
    maxes = np.max(scores, axis=1).reshape(N,1)
    scores -= maxes # shift by max(scores)
    
    # Softmax:
    numer = np.exp(scores) # (N,C)
    denom = np.sum(numer, axis=1).reshape(N,1) # this is for broadcasting, (N,1)
    softmax = numer / denom
    
    # Loss:
    loss = -np.log(softmax[np.arange(N), y]).sum()
    
    # Gradient:
    softmax[np.arange(N), y] -= 1 # subtract those x_i belongs to y_i classes (~ line 47)
    dW = np.dot(X.T, softmax)
    
    # Average:
    loss /= N
    dW /= N
    
    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

