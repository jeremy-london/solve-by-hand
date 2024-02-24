import numpy as np


# Define the ReLU activation function for array inputs
def relu(x):
    return np.maximum(0, x)


# Function to compute the output of a two-layer network with ReLU activations
def two_layer_network(W1, b1, W2, b2, x):
    # First layer operation: Hidden Layer
    h = relu(np.dot(W1, x) + b1)

    # Second layer operation: Output Layer
    y = relu(np.dot(W2, h) + b2)

    return h, y


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

x = np.array([2, 1, 3])  # Input vector

# Perform the network operation
h, y = two_layer_network(W1, b1, W2, b2, x)

# Calculate the parameters count
hidden_params = W1.shape[0] * (W1.shape[1] + 1)
output_params = W2.shape[0] * (W2.shape[1] + 1)
total_params = hidden_params + output_params

# Print the outcomes and parameter count
print("Hidden layer: (W1 * x + b1 → ReLU → h):\n", h)
print("Output Layer: (W2 * h + b2 → ReLU → y):\n", y)
print(
    "Total parameters in the network (hidden_params + output_params):\n",
    f"{hidden_params} + {output_params} = {total_params}",
)
