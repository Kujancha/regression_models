import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class Model:
    def __init__(self, X=None, y=None):
        self.X = X if X is not None else np.random.randn(100, 1)
        self.y = y if y is not None else self.X ** 2 + 3 * self.X + 0.5 + np.random.randn(100, 1)
        
        self.t0, self.t1 = 1, 10 \
        self.n_epochs = 100
        self.size_m = len(self.X)
        self.theta = None

    def learning_schedule(self, t: int) -> float:
        return self.t0 / (t + self.t1)

    def evaluation(self):
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        X_biased = np.c_[np.ones((self.size_m, 1)), X_scaled]
        
       \
        n_features = X_biased.shape[1]
        theta = np.random.randn(n_features, 1) * 0.01

        for epoch in range(self.n_epochs):
            for iteration in range(self.size_m):
                random_index = np.random.randint(self.size_m)
                Xi = X_biased[random_index:random_index + 1]
                yi = self.y[random_index:random_index + 1]
                
                gradient = 2 * Xi.T @ (Xi @ theta - yi)
                gradient = np.clip(gradient, -1, 1) \\
                
                alpha = self.learning_schedule(epoch * self.size_m + iteration)
                theta = theta - alpha * gradient

        self.theta = theta
        return self.theta

m = 100
X = 0.5 * 6 + np.random.randn(m, 1)
y = X ** 2 + 3 * X + 0.5 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

model = SGDRegressor(X_poly, y)
result = model.evaluation()
print("Learned theta (scaled):", result.ravel())