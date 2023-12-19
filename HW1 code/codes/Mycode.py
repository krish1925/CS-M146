import numpy as np
import random

random.seed(10)
np.random.seed(10)

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.theta
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        ### These two lines set the random seeds... you can ignore. #####
        random.seed(10)
        np.random.seed(10)
        #################################################################
        self.theta = np.random.standard_normal(self.dim)
    def get_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (n,1) containing the data.
        Returns:
         - X_out: an augmented training data as an mth degree feature vector 
         e.g. [1, x, x^2, ..., x^m], x \in X.
        """
        n,d = X.shape
        m = self.m
        X_out= np.zeros((n,m+1))
        if m==1:
            X_out[:, 0] = 1
            X_out[:, 1] = X[:, 0]
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out with each entry = [1, x, x^2,....,x^m]
            # ================================================================ #
            for i in range(m+1):
                X_out[:,i] = X[:,0]**i
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            pass
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: n x d array of training data.
        - y: n x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.theta containing the gradient of the loss with respect to self.theta 
        """
        loss = 0.0
        grad = np.zeros_like(self.theta) 
        m = self.m
        n,d = X.shape 

        y_pred = self.predict(X)
        loss = 0.5 * np.mean((y_pred - y) ** 2)

        if m==1:
            X_augmented = np.c_[np.ones(X.shape[0]), X]
            for j in range(self.theta.shape[0]):
                grad[j] = np.mean((y_pred - y) * X_augmented[:, j])
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            X_extended = self.get_poly_features(X)
            for j in range(self.theta.shape[0]):
                 grad[j] = np.mean((y_pred - y) * X_extended[:, j])
        return loss, grad
    
    def train_LR(self, X, y, alpha=1e-2, B=30, num_iters=10000) :
        """

        """
        ### These two lines set the random seeds... you can ignore. #####
        random.seed(10)
        np.random.seed(10)
        #################################################################
        self.theta = np.random.standard_normal(self.dim)
        loss_history = []
        n,d = X.shape
        for t in np.arange(num_iters):
            X_batch = None
            y_batch = None
            indices = np.arange(n)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            X_batch = X_shuffled[:B]
            y_batch = y_shuffled[:B]
            loss = 0.0
            grad = np.zeros_like(self.theta)

            loss, grad = self.loss_and_grad(X_batch, y_batch)
            self.theta -= alpha * grad
            # ================================================================ 
            loss_history.append(loss)
        return loss_history, self.theta
    
    def closed_form(self, X, y, lambda_red = 0.0):
        """
        Inputs:
        - X: n x 1 array of training data.
        - y: n x 1 array of targets
        Returns:
        - self.theta: optimal weights 
        """
        m = self.m
        n,d = X.shape
        X_bias = np.c_[np.ones((n, 1)), X]

        loss = 0
        if m==1:

            x_input = self.get_poly_features(X)
            self.theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_input), x_input)), np.matmul(np.transpose(x_input),y))
            loss, grad = self.loss_and_grad(X,y)

        else:

            x_input = self.get_poly_features(X)
            self.theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_input), x_input)), np.matmul(np.transpose(x_input),y))
            loss, grad = self.loss_and_grad(X,y)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.theta

    
    def predict(self, X):
        """
        Inputs:
        - X: n x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
        array of length n.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        if m == 1:
        
            X_augmented = np.c_[np.ones(X.shape[0]), X]
            y_pred = np.dot(X_augmented, self.theta).flatten()
        else:

            X_extended = self.get_poly_features(X)
            y_pred = np.dot(X_extended, self.theta).flatten()

        return y_pred