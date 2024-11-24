from abc import ABC, abstractmethod
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression

import utils

class Model(ABC):
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        # State
        self.trained = False
        
    @abstractmethod
    def train(self, X, y, **kwargs):
        return NotImplementedError("Subclasses should implement this method")
        
    @abstractmethod
    def evaluate(self, X, y):
        return NotImplementedError("Subclasses should implement this method")
        
    @abstractmethod
    def predict(self, X):
        return NotImplementedError("Subclasses should implement this method")

class SVM(Model):
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=1, coef0=0.0):
        super(self, Model).__init__()
        self.clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
        
    def train(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self.predict(X)
    
    def evaluate(self, X, y):
        preds = self.clf.predict(X)
        return (utils.compute_accuracy(y, preds), utils.compute_F1(y, preds))
    
    def predict(self, X):
        return self.clf.predict(X)
        
class LogisticRegression(Model):
    def __init__(self, rand_state):
        super(self, Model).__init__()
        self.rand_state = rand_state
        self.clf = LogisticRegression(random_state=rand_state)
        
    def train(self, X, y, lr=0.01, max_iters=1000):
        self.clf = self.clf.fit(X, y)
        return self.predict(X)
    
    def evaluate(self, X, y):
        preds = self.clf.predict(X)
        return (utils.compute_accuracy(y, preds), utils.compute_F1(y, preds))
    
    def predict(self, X):
        return self.clf.predict(X)

# The following models are inspired by https://arxiv.org/pdf/1511.06448
class NNMaxpool(Model):
    def __init__(self):
        super(self, Model).__init__()
        
    def train(self, X, y):
        pass
    
    def evaluate(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
class NNTempConv(Model):
    def __init__(self):
        super(self, Model).__init__()
        
    def train(self, X, y):
        pass
    
    def evaluate(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
class NNLSTM(Model):
    def __init__(self):
        super(self, Model).__init__()
        
    def train(self, X, y):
        pass
    
    def evaluate(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
class NNMixed(Model):
    def __init__(self):
        super(self, Model).__init__()
        
    def train(self, X, y):
        pass
    
    def evaluate(self, X, y):
        pass
    
    def predict(self, X):
        pass