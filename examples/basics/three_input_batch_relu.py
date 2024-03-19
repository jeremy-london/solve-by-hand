import numpy as np

# Given values for the two-layer network
W1 = np.array([[1, -1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
b1 = np.array([-5, 0, 1, -2])  # Biases for the hidden layer

W2 = np.array([[1, 1, -1, 0], [0, 0, 1, -1]])
b2 = np.array([0, 1])  # Biases for the output layer

# Define input batch
x_batch = np.array([[2, 1, 0], [1, 1, 1], [3, 0, 1]])

# Calculate the hidden layer's output before activation
h_z = np.dot(W1, x_batch) + b1.reshape(-1, 1)
# Apply ReLU activation to the hidden layer's output
h = np.maximum(0, h_z)

# Display the hidden layer's computation before and after ReLU activation
print("Hidden layer:\n", h_z)
print("ReLU Activated Hidden layer:\n", h)

# Compute the output layer's result before activation
y_z = np.dot(W2, h) + b2.reshape(-1, 1)
# Apply ReLU activation to the output layer's result
y = np.maximum(0, y_z)

# Display the output layer's computation before and after ReLU activation
print("Output Layer:\n", y_z)
print("ReLU Activated Output Layer:\n", y)
