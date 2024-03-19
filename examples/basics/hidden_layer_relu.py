import numpy as np

# Define the weight matrices for the hidden and output layers
W1 = np.array([[1, -1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
b1 = np.array([-5, 0, 1, -2])  # Hidden layer biases

W2 = np.array([[1, 1, -1, 0], [0, 0, 1, -1]])  # Output layer weights
b2 = np.array([0, 1])  # Output layer biases

x = np.array([2, 1, 3])  # Input vector

# Calculate the hidden layer's output before activation
h_z = np.dot(W1, x) + b1
# Apply ReLU activation to the hidden layer's output
h = np.maximum(0, h_z)

# Display the hidden layer's computation before and after ReLU activation
print("Hidden layer: (W1 * x + b1 → h_z):\n", h_z)
print("ReLU Activated Hidden layer: (W1 * x + b1 → ReLU → h):\n", h)

# Compute the output layer's result before activation
y_z = np.dot(W2, h) + b2
# Apply ReLU activation to the output layer's result
y = np.maximum(0, y_z)

# Display the output layer's computation before and after ReLU activation
print("Output Layer: (W2 * h + b2 → y_z):\n", y_z)
print("ReLU Activated Output Layer: (W2 * h + b2 → ReLU → y):\n", y)

# Calculate the total number of parameters in the network
hidden_params = W1.shape[0] * (W1.shape[1] + 1)  # Parameters in the hidden layer
output_params = W2.shape[0] * (W2.shape[1] + 1)  # Parameters in the output layer
total_params = hidden_params + output_params  # Sum of all parameters

# Print the total parameter count for the network
print(
    "Total parameters in the network (hidden_params + output_params):\n",
    f"{hidden_params} + {output_params} = {total_params}",
)
