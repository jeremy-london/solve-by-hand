import numpy as np


# Applying ReLU activation to the expert outputs
def relu(x):
    return np.maximum(0, x)


# Using softmax function to determine the weights for each expert
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


# Define the inputs
X = np.array([[2, 1, 3], [3, 1, 2]])

# Define gate network weights and bias
gate_weights = np.array([[1, 1, 0], [0, 1, 1]])
gate_bias = np.array([0, 0])

# Define expert weights and bias
expert1_weights = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [-1, 0, 1]])
expert2_weights = np.array([[0, 1, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1]])
expert_bias = np.array([0, 0, 0, 0])

# Gate network processing
# Applying the gate network to the inputs
gate_outputs = np.dot(X, gate_weights.T) + gate_bias
gate_probabilities = softmax(gate_outputs)

# Processing by Expert 2 and Expert 1
# Expert 2 processes the first input (X1)
expert2_output_X1 = np.dot(X[0], expert2_weights.T) + expert_bias

# Expert 1 processes the second input (X2)
expert1_output_X2 = np.dot(X[1], expert1_weights.T) + expert_bias

# ReLU activation function
Y1 = relu(expert2_output_X1)
Y2 = relu(expert1_output_X2)

print("Gate Outputs:\n", gate_outputs)
print("Gate Probabilities:\n", gate_probabilities)

print("Expert 2 Output (X1):\n", expert2_output_X1)
print("ReLU Expert 2 Output (X1):\n", Y1)

print("Expert 1 Output (X2):\n", expert1_output_X2)
print("ReLUExpert 1 Output (X2):\n", Y2)
