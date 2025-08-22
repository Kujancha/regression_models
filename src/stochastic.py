import numpy as np
import matplotlib.pyplot as plt


class stochastic:
    def __init__(self):
        
       
        
        self.X = 2 * np.random.randn(100,1)
        self.y = 4 + 2 * self.X + np.random.randn(100,1)

        self.t0, self.t1 = 5,50 # learning schedule parameters
        self.n_epochs = 100 # no. of iterations

        self.size_m = len(self.X)
        self.theta = None
    
    def learning_schedule(self, t:int)->int:
        return self.t0 / (t + self.t1)
    
    def evalutation(self):
        X_biased = np.c_[np.ones((100,1)), self.X]
        theta = np.random.rand(2,1)  # initialize a random theta

        for epoch in range(self.n_epochs):
            for iteration in range(self.size_m):
                random_index = np.random.randint(self.size_m)
                Xi = X_biased[random_index: random_index + 1] # array splicing cos we need to still perform matrix operations
                yi = self.y[random_index:random_index+1]
                gradient = 2 * Xi.T@(Xi@theta - yi)
                alpha = self.learning_schedule(epoch * self.size_m + iteration)   # this is the learning rate
                theta = theta - alpha * gradient
                
        self.theta = theta
        return self.theta
    
    
    def line_function(self, x):
        intercept, slope = self.theta.ravel()
        return slope * x + intercept
    
    
            
    def finalize(self):
        plt.scatter(self.X, self.y, label="Data")   
        self.evalutation()
        
        
        X_sorted = np.sort(self.X, axis=0)
        plt.plot(X_sorted, self.line_function(X_sorted), color='red', label="Fitted line")
        
        print(f"The predicted parameters are: {self.theta.ravel()}")
        plt.legend()
        plt.show()
          
                


def main():
    sto = stochastic()
    sto.finalize()
    
    
if __name__ == "__main__":
    main()









