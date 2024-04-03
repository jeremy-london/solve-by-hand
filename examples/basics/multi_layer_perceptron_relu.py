import numpy as np

# Given values for the two-layer network
W1 = np.array([
    [0, 0, 1],  # Hidden Neuron 1 weights
    [0, 1, 0],  # Hidden Neuron 2 weights
    [1, 0, 0],  # Hidden Neuron 3 weights
    [1, 1, 0],  # Hidden Neuron 4 weights
    [0, 1, 1]  # Hidden Neuron 5 weights
])

W2 = np.array([
    [1, 1, -1, 0, 0],  # Hidden Neuron 1 weights
    [0, 0, 1, 1, -1]  # Hidden Neuron 2 weights
])

W3 = np.array([
    [1, 1],  # Hidden Neuron 1 weights
    [1, -1],  # Hidden Neuron 2 weights
    [1, 2]  # Hidden Neuron 3 weights
])

W4 = np.array([
    [1, -1, 0],  # Hidden Neuron 1 weights
    [0, -1, 1]  # Hidden Neuron 2 weights
])

W5 = np.array([
    [0, 1],  # Hidden Neuron 1 weights
    [1, 0]  # Hidden Neuron 2 weights
])

W6 = np.array([
    [1, -1],  # Hidden Neuron 1 weights
    [1, 1]  # Hidden Neuron 2 weights
])

W7 = np.array([
    [1, -1],  # Output Neuron 1 weights
])

x_batch = np.array([
    [3, 5],  # Input features for x1
    [4, 4],  # Input features for x2
    [5, 3]  # Input features for x3
])


# Define the ReLU activation function for array inputs
def relu(x):
    return np.maximum(0, x)


# Multi-layer perceptron function
def multi_layer_perceptron(W1, W2, W3, W4, W5, W6, W7, x_batch):
    # Layer 1
    h1_before_activation = np.dot(W1, x_batch)
    h1_after_activation = relu(h1_before_activation)

    # Layer 2
    h2_before_activation = np.dot(W2, h1_after_activation)
    h2_after_activation = relu(h2_before_activation)

    # Layer 3
    h3_before_activation = np.dot(W3, h2_after_activation)
    h3_after_activation = relu(h3_before_activation)

    # Layer 4
    h4_before_activation = np.dot(W4, h3_after_activation)
    h4_after_activation = relu(h4_before_activation)

    # Layer 5
    h5_before_activation = np.dot(W5, h4_after_activation)
    h5_after_activation = relu(h5_before_activation)

    # Layer 6
    h6_before_activation = np.dot(W6, h5_after_activation)
    h6_after_activation = relu(h6_before_activation)

    # Output layer
    y_before_activation = np.dot(W7, h6_after_activation)

    return (
        h1_after_activation,
        h2_after_activation,
        h3_after_activation,
        h4_after_activation,
        h5_after_activation,
        h6_after_activation,
        y_before_activation,
    )


# Call the multi-layer perceptron function
results = multi_layer_perceptron(W1, W2, W3, W4, W5, W6, W7, x_batch)

# Display the results
for i, result in enumerate(results, start=1):
  print(f"Layer {i} output:\n{result}\n")
