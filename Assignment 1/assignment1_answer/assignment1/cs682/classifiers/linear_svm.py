import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Also need to divide dW
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    
    N, C = np.shape(X)[0], np.shape(W)[1]
    
    scores = np.dot(X, W) # (N,C)
    # Selecting specific column index per row given y (This will return list of W_yi x_i)
    correct_class_scores = scores[np.arange(N), y].reshape(N, 1) # This will help for broadcasting 
    margins = np.maximum(0, scores - correct_class_scores + 1) # note that we also include the case where j = y_i
    # Filter out those correct classes
    margins[np.arange(N), y] = 0
    loss = np.sum(margins, axis=1)
    
    # Sum over all L_i
    loss = loss.sum()
    # Average the loss
    loss /= N
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    
    # Filter out all non-zero margins, turning margins into 0-1 matrix:
    margins[margins > 0] = 1
    # Update margins at column indices according y by row sum over all # of x that belongs to different classes
    margins[np.arange(N), y] -= np.sum(margins, axis=1)
    
    dW = np.dot(X.T, margins)
    dW /= N # Average over num_trains
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
