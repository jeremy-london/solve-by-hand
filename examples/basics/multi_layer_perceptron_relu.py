import numpy as np

# Define individual input vectors as column vectors
x1 = np.array([[3], [4], [5]])
x2 = np.array([[5], [4], [3]])

x_batch = np.hstack((x1, x2))

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

def relu(x):
    return np.maximum(0, x)


# Multi-layer perceptron function
def multi_layer_perceptron(W1, W2, W3, W4, W5, W6, W7, x_batch):
    h1_activation = relu(np.dot(W1, x_batch))
    h2_activation = relu(np.dot(W2, h1_activation))
    h3_activation = relu(np.dot(W3, h2_activation))
    h4_activation = relu(np.dot(W4, h3_activation))
    h5_activation = relu(np.dot(W5, h4_activation))
    h6_activation = relu(np.dot(W6, h5_activation))
    y_before_activation = np.dot(W7, h6_activation)

    return (
        h1_activation,
        h2_activation,
        h3_activation,
        h4_activation,
        h5_activation,
        h6_activation,
        y_before_activation,
    )


# Call the multi-layer perceptron function
results = multi_layer_perceptron(W1, W2, W3, W4, W5, W6, W7, x_batch)

# Display the results
for i, result in enumerate(results, start=1):
  print(f"Layer {i} output:\n{result}\n")
