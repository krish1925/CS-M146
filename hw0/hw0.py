import numpy as np
import matplotlib.pyplot as plt

mean = [0,0]
cov_matrix = np.array([[1,-0.5],[-0.5,1]])
samples = np.random.multivariate_normal(mean, cov_matrix,1000)

x1 = samples[:,0]
x2 = samples[:,1]

plt.scatter(x1, x2, s=10, alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of 2D Gaussian Samples')
plt.grid(True)
plt.show()
