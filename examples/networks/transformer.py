import numpy as np

# Define input features
X = np.array([[5, 0, 1], [6, 2, 0], [0, 4, 1], [7, 0, 1], [0, 3, 0]])

# Define the Query-Key (QK) attention weight matrix
A = np.array(
    [
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]
)

# Apply attention weighting
Z = np.dot(A.T, X)  # Multiply the input features by the attention weight matrix

print("Output after attention weighting:\n", Z.T)

# Define weights and biases for the position-wise feed-forward network (FFN)
weights_1 = np.array([[1, -1, 0], [1, 1, 0], [0, 1, 1], [-1, 1, 1]])
bias_1 = np.array([1, 0, 1, 0])

weights_2 = np.array([[1, 0, 0, -1], [0, 1, 1, 0], [0, 0, 1, -1]])
bias_2 = np.array([0, 0, 1])

# FFN: First Layer
FFN_1_output = np.dot(Z, weights_1.T) + bias_1
print("Output after first FFN layer:\n", FFN_1_output.T)
FFN_1_output = np.maximum(0, FFN_1_output)  # Apply ReLU
print("Output after ReLU activation:\n", FFN_1_output.T)

# FFN: Second Layer
FFN_2_output = np.dot(FFN_1_output, weights_2.T) + bias_2
print("Output after Transformer Block:\n", FFN_2_output.T)
