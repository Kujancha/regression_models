import numpy as np
import matplotlib.pyplot as plt

m = 100
X = 6  * np.random.rand(m,1) - 3
y = X ** 2 + 3 * X + 0.5 + np.random.rand(m,1)

plt.scatter(X,y)
plt.show() 

