import numpy as np


def sigmoid (x):
	return 1 / (1 + np.exp(-x))


def normalize (X):
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	return (X - mean) / std


class my_LogisticRegression:
	def __init__ (self, alpha=0.01, iterations=10000):
		self.alpha = alpha
		self.iterations = iterations
		self.theta = None
	
	def cost_function (self, y, h):
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
	
	def fit (self, X, y, cost=False):
		self.theta = np.zeros(X.shape[1])
		for i in range(self.iterations):
			z = np.dot(X, self.theta)
			h = sigmoid(z)
			g = np.dot(X.T, (h - y)) / len(y)
			self.theta -= self.alpha * g
			if cost and i % 1000 == 0:
				print(f"Iteration: {i}, Cost: {self.cost_function(y, h)}")
		
		return self.theta
	
	def predict (self, X):
		p = sigmoid(np.dot(X, self.theta))
		p[p >= 0.5] = 1
		p[p < 0.5] = 0
		return p
	
	def accuracy (self, y_true, y_pred):
		return (y_true == y_pred).mean()
	
	def precision (self, y_true, y_pred):
		true_positive = np.sum((y_pred == 1) & (y_true == 1))
		predicted_positive = np.sum(y_pred == 1)
		return true_positive / predicted_positive if predicted_positive != 0 else 0
	
	def recall (self, y_true, y_pred):
		true_positive = np.sum((y_pred == 1) & (y_true == 1))
		actual_positive = np.sum(y_true == 1)
		return true_positive / actual_positive if actual_positive != 0 else 0
	
	def f1_score (self, y_true, y_pred):
		prec = self.precision(y_true, y_pred)
		rec = self.recall(y_true, y_pred)
		return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
