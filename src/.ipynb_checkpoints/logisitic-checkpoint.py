import numpy as np




class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_biased = np.c_[np.ones((len(X), 1)), self.X]
        self.n_samples, self.n_features = self.X_biased.shape
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        
    
    def fit(self, alpha = 0.1, max_ite = 100):
        
        theta = np.zeros(self.n_features)
        
        for i in range(max_ite):
            z = self.X_biased @ theta
            y_pred = self.sigmoid(z) - self.y
            gradient = self.X_biased.T @ y_pred
            gradient  /= self.n_samples
            theta = theta - alpha * gradient
            
        self.theta = theta
        return theta
    
    def predict_probability(self, test_set):
        test_set_biased = np.c_[np.ones((len(test_set), 1)), test_set]
        return self.sigmoid(test_set_biased @ self.theta)
    
    def predict(self,X, threshold = 0.5):
        return (self.predict_probability(X) > threshold).astype(int)

    def confusion_matrix(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        return np.array([[TN, FP],
                        [FN, TP]])        
    

