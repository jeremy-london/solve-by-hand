import numpy as np

# Mini-batch input
X = np.array([[1, 0, 2], [0, 3, 1], [3, 1, 0], [0, 1, 2]])

# Linear layer weights and bias
weights = np.array([[1, 0, 1], [1, 1, 0], [0, 2, -1]])
bias = np.array([0, -1, 0])

# Step 2: Apply the linear layer
linear_output = np.dot(X, weights.T) + bias
print("Linear output:\n", linear_output.T)

# Step 3: Apply the ReLU activation function
relu_output = np.maximum(0, linear_output.T)
print("ReLU output:\n", relu_output)

# Step 4: Compute batch statistics}
sums = np.round(np.sum(relu_output, axis=1).reshape(-1, 1))  # Sum across each feature
mean = np.round(np.mean(relu_output, axis=1).reshape(-1, 1))  # Mean across each feature
variance = np.round(
    np.var(relu_output, axis=1).reshape(-1, 1)
)  # Variance across each feature
std_dev = np.round(
    np.sqrt(variance + 1e-8).reshape(-1, 1)
)  # Std deviation across each feature

print(f"Sum: \n{sums}\nMean: \n{mean}\nVariance: \n{variance}\nStd Dev: \n{std_dev}")

# Step 5: Normalize the batch to mean = 0
normalized_output = relu_output - mean
print("Normalized output:\n", normalized_output)

# Step 6: Scale the batch to variance = 1
scaled_output = normalized_output / std_dev
print("Scaled output:\n", scaled_output)

# Step 7: Scale and shift
# Trainable parameters (gamma and beta)
gamma = np.array([2, 3, -1]).reshape(-1, 1)  # Example values for gamma
beta = np.array([0, 0, 1]).reshape(-1, 1)  # Example values for beta


# Apply scale and shift transformation
scale_and_shift_output = gamma * scaled_output + beta
print("Scaled and shifted output:\n", scale_and_shift_output)

# Output ready to be passed to the next layer
next_layer_input = np.round(scale_and_shift_output, 1)
print("Next layer input:\n", next_layer_input)
