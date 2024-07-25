import numpy as np
from typing import Union


class my_LogisticRegression:
    """
    Logistic Regression model.

    :param alpha: Learning rate for gradient descent.
    :type alpha: float
    :param iterations: Number of iterations for gradient descent.
    :type iterations: int
    """
    
    def __init__ (self, alpha: float = 0.01, iterations: int = 10000):
        """
        Initialize the Logistic Regression model.

        :param alpha: Learning rate for gradient descent.
        :type alpha: float
        :param iterations: Number of iterations for gradient descent.
        :type iterations: int
        """
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None
    
    def cost_function (self, y: np.ndarray, h: np.ndarray) -> float:
        """
        Compute the cost function for logistic regression.

        :param y: True labels.
        :type y: np.ndarray
        :param h: Predicted probabilities.
        :type h: np.ndarray

        :return: Cost value.
        :rtype: float
        :raises ValueError: If y and h do not have the same length.
        """
        if len(y) != len(h):
            raise ValueError("The length of y and h must be the same.")
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    @staticmethod
    def sigmoid (x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.

        :param x: Input values.
        :type x: np.ndarray

        :return: Sigmoid values.
        :rtype: np.ndarray
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def normalize (X: np.ndarray) -> np.ndarray:
        """
        Normalize the feature matrix X.

        :param X: Feature matrix.
        :type X: np.ndarray

        :return: Normalized feature matrix.
        :rtype: np.ndarray
        :raises ValueError: If X has less than 2 dimensions.
        """
        if X.ndim < 2:
            raise ValueError("X must be a 2D array.")
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std
    
    def fit (self, X: np.ndarray, y: np.ndarray, cost: bool = False) -> np.ndarray:
        """
        Fit the Logistic Regression model to the training data.

        :param X: Training feature matrix.
        :type X: np.ndarray
        :param y: Training labels.
        :type y: np.ndarray
        :param cost: Whether to print cost during training.
        :type cost: bool

        :return: Fitted model parameters (weights).
        :rtype: np.ndarray
        :raises ValueError: If X and y have incompatible shapes.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2:
            y = y.flatten()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same.")
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.iterations):
            z = np.dot(X, self.theta)
            h = my_LogisticRegression.sigmoid(z)
            g = np.dot(X.T, (h - y)) / len(y)
            self.theta -= self.alpha * g
            if cost and i % 1000 == 0:
                print(f"Iteration: {i}, Cost: {self.cost_function(y, h)}")
        
        return self.theta
    
    def predict (self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given feature matrix X.

        :param X: Feature matrix.
        :type X: np.ndarray

        :return: Predicted labels.
        :rtype: np.ndarray
        :raises ValueError: If X has fewer columns than the number of features in the model.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.theta):
            raise ValueError("The number of features in X must match the number of features used during training.")
        
        p = my_LogisticRegression.sigmoid(np.dot(X, self.theta))
        p[p >= 0.5] = 1
        p[p < 0.5] = 0
        return p
    
    def accuracy (self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy of the predictions.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray

        :return: Accuracy value.
        :rtype: float
        :raises ValueError: If y_true and y_pred do not have the same length.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("The length of y_true and y_pred must be the same.")
        return (y_true == y_pred).mean()
    
    def precision (self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the precision of the predictions.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray

        :return: Precision value.
        :rtype: float
        :raises ValueError: If y_true and y_pred do not have the same length.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("The length of y_true and y_pred must be the same.")
        
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive != 0 else 0
    
    def recall (self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the recall of the predictions.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray

        :return: Recall value.
        :rtype: float
        :raises ValueError: If y_true and y_pred do not have the same length.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("The length of y_true and y_pred must be the same.")
        
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive != 0 else 0
    
    def f1_score (self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the F1 score of the predictions.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray

        :return: F1 score value.
        :rtype: float
        :raises ValueError: If y_true and y_pred do not have the same length.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("The length of y_true and y_pred must be the same.")
        
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
