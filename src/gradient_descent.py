import numpy as np
import matplotlib.pyplot as plt


# generate a testing data for instance
X = 2 * np.random.rand(100,1)  # 100 x 1 random nu between 0 and 1
y = 4 + 3 * X + np.random.randn(100, 1)


# adding bias term for intercept
X_biased = np.c_[np.ones((100,1)), X] # this will like append 1 to our sample

theta = np.random.randn(2, 1)

alpha = 0.01  # learning_rate
iterations = np.size(X)

plot_every = 10


plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Data points")


for i in range(iterations):
    y_pred = X_biased @ theta
    error = y_pred - y
    gradients = (2 / len(X_biased)) * X_biased.T @ error
    theta = theta - alpha * gradients
    
    
    # this is to plot
    
    if i % plot_every == 0:
        x_line = np.array([[0], [2]])
        x_line_b = np.c_[np.ones((2, 1)), x_line]
        y_line = x_line_b @ theta
        plt.plot(x_line, y_line, label=f"Step {i}")


x_line = np.array([[0], [2]])
x_line_b = np.c_[np.ones((2, 1)), x_line]
y_line = x_line_b @ theta
plt.plot(x_line, y_line, color="black", linewidth=2, label="Final fit")

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Gradient Descent for Linear Regression")
plt.show()

print("Estimated parameters (intercept, slope):", theta.ravel())
