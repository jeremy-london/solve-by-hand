import numpy as np

# Define the matrices
A = np.array([[1, 1], [-1, 1]])
B = np.array([[1, 5, 2], [2, 4, 2]])

# Perform matrix multiplication
C = np.dot(A, B)

# Print the outcomes and parameter count
print("Matrix A * Matrix B:\n", C)
