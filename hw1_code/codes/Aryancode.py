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
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out with each entry = [1, x]
            # ================================================================ #
            for i in range(n):
                xi = X[i,0]
                X_out[i,0] = 1
                X_out[i, 1] = xi
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out with each entry = [1, x, x^2,....,x^m]
            # ================================================================ #
            for i in range(len(X)):
                for j in range(m+1):
                    X_out[i][j] = np.power(X[i], j)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            #pass
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
        if m==1:
            y_predict = self.predict(X)
            loss = 0.5 * np.mean((y_predict - y) ** 2)
            X_ext = np.c_[np.ones(X.shape[0]), X]
            for j in range(self.theta.shape[0]):
                grad[j] = np.mean((y_predict - y) * X_ext[:, j])
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            y_predict = self.predict(X)
            X_polynomial = self.get_poly_features(X)
            for i in range(self.theta.shape[0]):
                grad[i] = np.mean((y_predict - y) * X_polynomial[:, i])
                if(i != 0):
                    grad[i] += 2*self.reg*self.theta[i]
            loss = 0.5 * np.mean((y_predict - y) ** 2)
        return loss,grad
    
    def train_LR(self, X, y, alpha=1e-2, B=30, num_iters=10000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares mini-batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (n,d), features
         - y         -- numpy array of shape (n,), targets
         - alpha     -- float, learning rate
         -B          -- integer, batch size
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.theta: optimal weights 
        """
        ### These two lines set the random seeds... you can ignore. #####
        random.seed(10)
        np.random.seed(10)
        #################################################################
        self.theta = np.random.standard_normal(self.dim)
        loss_history = []
        n,d = X.shape
        shuff = np.column_stack((X,y))
        for t in np.arange(num_iters):
            X_batch = None
            y_batch = None

            np.random.shuffle(shuff)
            X_batch = shuff[:B, :-1]
            y_batch = shuff[:B,-1].reshape((-1,1))

            loss = 0.0
            grad = np.zeros_like(self.theta)
            # ================================================================ #
            # YOUR CODE HERE: 
            # evaluate loss and gradient for batch data
            # save loss as loss and gradient as grad
            # update the weights self.theta
            # ================================================================ #
            loss,grad = self.loss_and_grad(X_batch,y_batch)
            self.theta -= alpha*grad
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss_history.append(loss)
        return loss_history, self.theta
    
    def closed_form(self, X, y):
        """
        Inputs:
        - X: n x 1 array of training data.
        - y: n x 1 array of targets
        Returns:
        - self.theta: optimal weights 
        """
        m = self.m
        n,d = X.shape
        loss = 0
        # if m==1:
        #     # ================================================================ #
        #     # YOUR CODE HERE:
        #     # obtain the optimal weights from the closed form solution 
        #     # ================================================================ #
        #     #print(X.shape)
        #     A = np.hstack((X, np.ones((X.shape[0],1))))
        #     #print(A.shape)
        #     self.theta = np.linalg.inv(A.T @ A) @ A.T @ y
        #     y_pred = A @ self.theta
        #     loss = np.mean((y_pred-y)**2)
        #     # ================================================================ #
        #     # END YOUR CODE HERE
        #     # ================================================================ #
        # else:
            # ================================================================ #
            # YOUR CODE HERE:
            # Extend X with get_poly_features().
            # Predict the targets of X.
            # ================================================================ #
        X_in = self.get_poly_features(X)
        self.theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_in),X_in)),np.matmul(np.transpose(X_in), y))
        loss, grad = self.loss_and_grad(X,y)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.theta
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
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            X_ext = np.c_[np.ones(X.shape[0]), X]
            y_pred = np.dot(X_ext, self.theta).flatten()
        else:

            X_polynomial = self.get_poly_features(X)
            y_pred = np.dot(X_polynomial, self.theta).flatten()
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred
