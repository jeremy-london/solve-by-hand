import numpy as np


# Define the activation function
def relu(x):
    return np.maximum(0, x)


# Initial hidden state
h = np.array([0, 0])

# Input sequence
X = np.array([3, 4, 5, 6])

# Parameters
A = np.array([[1, -1], [1, 1]])
B = np.array([[1], [2]])
C = np.array([-1, 1])

# Output list, ensuring it's a list of scalars
Y = []

# Apply parameters to input sequence
for i, xi in enumerate(X):
    # Update for parameters
    x = np.dot(A, h) + xi * B.flatten()
    print(f"x{i} before ReLU:\n", x)

    # Adjusted to ensure proper broadcasting
    h = relu(x)
    print(f"x{i} after ReLU:\n", h)

    # Calculate output as a scalar
    yi = np.dot(C, h)  # This should now be a scalar value
    Y.append(yi)  # Append the scalar yi to the list Y

print("Y:\n", Y)
