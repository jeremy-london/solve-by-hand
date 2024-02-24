import numpy as np


# Define the ReLU activation function for array inputs
def relu(x):
    return np.maximum(0, x)


# Function to compute the output of a two-layer network with ReLU activations
def two_layer_network(W1, b1, W2, b2, x_batch):
    # First layer operation: Hidden Layer
    # Calculate the hidden layer outputs before activation
    # This is the dot product of W1 with each input vector plus the hidden bias
    # Adjusting for batch input - b1 reshaped for broadcasting
    h_before_activation = np.dot(W1, x_batch) + b1.reshape(-1, 1)

    # Apply the ReLU activation function to the hidden layer outputs
    h_after_activation = relu(h_before_activation)

    # Print the hidden layer outputs before and after applying the ReLU activation function
    print("Hidden layer outputs before ReLU activation:\n", h_before_activation)
    print("Hidden layer outputs after ReLU activation:\n", h_after_activation)

    # Second layer operation: Output Layer
    # This is the dot product of W2 with each ReLU activated hidden vector plus the output bias
    # Adjusting for batch input - b2 reshaped for broadcasting
    # y = relu(np.dot(W2, h_after_activation) + b2.reshape(-1, 1))
    y_before_activation = np.dot(W2, h_after_activation) + b2.reshape(-1, 1)

    # Apply the ReLU activation function to the hidden layer outputs
    y_after_activation = relu(y_before_activation)

    # Print the output layer outputs before and after applying the ReLU activation function
    print("Output layer outputs before ReLU activation:\n", y_before_activation)
    print("Output layer outputs after ReLU activation:\n", y_after_activation)

    return h_after_activation, y_after_activation


# Given values for the two-layer network
W1 = np.array(
    [
        [1, -1, 1],  # Hidden Neuron 1 weights
        [1, 1, 0],  # Hidden Neuron 2 weights
        [0, 1, 1],  # Hidden Neuron 3 weights
        [1, 0, 1],  # Hidden Neuron 4 weights
    ]
)
b1 = np.array([-5, 0, 1, -2])  # Biases for the hidden layer

W2 = np.array(
    [[1, 1, -1, 0], [0, 0, 1, -1]]  # Output Neuron 1 weights  # Output Neuron 2 weights
)
b2 = np.array([0, 1])  # Biases for the output layer

x_batch = np.array(
    [
        [2, 1, 0],  # Input features for x1
        [1, 1, 1],  # Input features for x2
        [3, 0, 1],  # Input features for x1
    ]
)

# Perform the network operation
h, y = two_layer_network(W1, b1, W2, b2, x_batch)
