import numpy as np

class GradientDescent:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_biased = np.c_[np.ones((len(X), 1)), X]
        self.n_samples, self.n_features = self.X_biased.shape
        
    def BatchGradientDescent(self, alpha=0.1, max_iterations=100):
        theta = np.zeros(self.n_features)
        
        for i in range(max_iterations):
            gradient = self.X_biased.T @ (self.X_biased @ theta - self.y)
            gradient /= self.n_samples
            theta = theta - alpha * gradient
        
        return theta
    
    def StochasticGradientDescent(self, t0=5, t1=50, n_epochs=50):
        theta = np.zeros(self.n_features)
     
        def learning_rate(t):
            return t0 / (t + t1)
         
        for epoch in range(n_epochs):
            for i in range(self.n_samples):
                random_index = np.random.randint(self.n_samples)
                xi = self.X_biased[random_index:random_index+1]  
                yi = self.y[random_index]
                
                gradient = xi.T @ (xi @ theta - yi)
                eta = learning_rate(epoch)
                theta = theta - eta * gradient.flatten()
                
                
                
        return theta
