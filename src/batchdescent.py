import numpy as np


np.random.seed(69)


X = 2 * np.random.randn(100,1)
size_m = len(X)

y = 4 + 2 * X + np.random.randn(100,1)


X_biased = np.c_[np.ones((100,1)), X]
theta = np.random.rand(2,1)



alpha = 0.1 #learning rate
n_epochs = 100 # no. of iterations

#each instance of gradient descent is called an epoch
for epoch in range(n_epochs):
    y_prediction = X_biased @ theta
    error = y_prediction - y
    gradient = (2/size_m) * X_biased.T @ error
    theta = theta - alpha * gradient
    

print(theta)