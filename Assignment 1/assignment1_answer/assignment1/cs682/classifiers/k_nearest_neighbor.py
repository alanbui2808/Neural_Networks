import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        if num_loops == 0:
          dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
          dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
          dists = self.compute_distances_two_loops(X)
        else:
          raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            test_image = X[i, :] 
            for j in range(num_train):
                '''
                TODO:
                Compute the l2 distance between the ith test point and the jth
                training point, and store the result in dists[i, j]. You should
                not use a loop over dimension.
                '''
                training_image = self.X_train[j, :]
                dists[i, j] = np.linalg.norm(test_image - training_image)

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            
            # We can broadcast self.X_train - X[i,:] since the rows are pairwise similar in dimension
            dists[i, :] = np.linalg.norm(self.X_train - X[i,:], axis = 1)
            
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        '''
        Using formula: (x - y)^2 = x.Tx - 2x.Ty + y.Ty
        
        We let X = X_test and Y = X_train
        
        We have: 
        (1). Compute x_i.T x_i for i: 0 --> 500, this results in (1,500) array (vector). Call this as X^2
        (2). Compute y_j.T y_j for j: 0 --> 5000, this resultsin (1,5000) array (vector). Call this as Y^2
        (3). Compute x_i.T y_j for all i and j: We can compute this by XY.T, this results in (500, 5000) matrix
        
        We can broadcast: X^2 + Y^2 if we reshape X^2 into (500,1). Which results in (500, 5000) matrix:
        [(x_1.T x_1) + (y_1.T y_1) , (x_1.T x_1) + (y_2.T y_2), ..., (x_1.T x_1) + (y_5000.T y_5000)
            .
            .
        (x_500.T x_500) + (y_1.T y_1) , (x_500.T x_500) + (y_2.T y_2), ..., (x_500.T x_500) + (y_5000.T y_5000)]
        
        We can finally compute: XX + YY - 2X.TY since the dimensions are all corect.
       
        '''
        X, Y = X, self.X_train
        
        XX = np.sum(X**2, axis = 1)
        YY = np.sum(Y**2, axis = 1) # (1, 5000)
        XX = np.reshape(XX, (np.shape(XX)[0], 1)) # (500, 1)
        XY = np.dot(X, Y.T) # (500, 5000)
        
        # XX + YY gets broadcast as XX will stretch (duplicate) its column by 5000 and YY will stretch (duplicate) its row by 500
        dists = np.sqrt(XX + YY - 2*XY)
    
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            
            distances = dists[i,:]
            
            # K-nearest neighbors
            if k > len(distances):
                raise ValueError('Invalid value of k since k > {} (num_training)'.format(self.X_train.shape[0]))
                
            k_nearest_neighbors = np.argsort(distances)[:k]
            
            # Fill the labels given y_train and k_nearest_neighbors (indices of k-nearest images)
            closest_y = np.take(self.y_train, k_nearest_neighbors)
            
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            
            labels_frequencies = np.bincount(closest_y)
            y_pred[i] = labels_frequencies.argmax()
            
            #########################################################################
            #                           END OF YOUR CODE                            # 
            #########################################################################

        return y_pred

