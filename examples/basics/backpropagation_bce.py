import numpy as np

# Activation function: Rectified Linear Unit (ReLU)
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU function
def relu_derivative(x):
    return (x > 0).astype(x.dtype)

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
# Input vector
X = np.array([[2], [1], [3]])

# Layer 1 parameters (weights and biases)
W1 = np.array([[1, -1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
b1 = np.array([-5, 0, 1, -2])

# Forward pass through Layer 1
z1 = np.dot(W1, X) + b1[:, None]
a1 = relu(z1)

# Layer 2 parameters
W2 = np.array([[1, -1, 1, 0], [0, 1, -1, 1]])
b2 = np.array([0, 3])

# Forward pass through Layer 2
z2 = np.dot(W2, a1) + b2[:, None]
a2 = relu(z2)

# Layer 3 parameters
W3 = np.array([[2, 0], [0, 2], [1, 1]])
b3 = np.array([-1, -5, -7])

# Forward pass through Layer 3
z3 = np.dot(W3, a2) + b3[:, None]

# Apply sigmoid activation to the output
Y_pred = sigmoid(z3)

# Target values
Y_target = np.array([[0], [1], [0]])

# Compute the BCE loss
epsilon = 1e-15  # To prevent log(0)
Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
loss = - (Y_target * np.log(Y_pred_clipped) + (1 - Y_target) * np.log(1 - Y_pred_clipped))
loss_total = np.mean(loss)

# Begin backpropagation

# Gradient of the loss with respect to z3
dL_dz3 = Y_pred - Y_target

# Predictions and target values
Y_pred = np.array([0.5, 0.5, 0])
Y_target = np.array([0, 1, 0])

# Begin backpropagation

# Compute the gradient of the loss with respect to z3 (Layer 3 pre-activation)
dL_dz3 = Y_pred - Y_target

# Compute the gradients of the loss with respect to Layer 3 weights and biases
dL_dW3 = np.dot(dL_dz3.reshape(-1, 1), a2.T)
dL_db3 = dL_dz3

# Compute the gradient of the loss with respect to a2 (Layer 2 activation)
dL_da2 = np.dot(W3.T, dL_dz3)

# Compute the gradient of the loss with respect to z2 (Layer 2 pre-activation)
dL_dz2 = dL_da2

# Compute the gradients of the loss with respect to Layer 2 weights and biases
dL_dW2 = np.dot(dL_dz2.reshape(-1, 1), a1.T)
dL_db2 = dL_dz2

# Compute the gradient of the loss with respect to a1 (Layer 1 activation)
dL_da1 = np.dot(W2.T, dL_dz2)

# Apply the derivative of ReLU to the backpropagated gradient
# to compute the gradient of the loss with respect to z1 (Layer 1 pre-activation)
dL_dz1 = dL_da1 * relu_derivative(z1.ravel())

# Compute the gradients of the loss with respect to Layer 1 biases
# This is the gradient of the loss with respect to z1 itself since we have a single sample
dL_db1 = dL_dz1

# Compute the gradients of the loss with respect to Layer 1 weights
# by outer product of the input X and dL_dz1
dL_dW1 = np.dot(X, dL_dz1.reshape(1, -1))

# Print the gradients for each layer to verify the backpropagation process
print("Layer 3 gradients:")
print(" z3:\n", z3.T)
print(" ∂L/∂b3:\n", dL_db3)
print(" ∂L/∂W3:\n", dL_dW3.T)

print("Layer 2 gradients:")
print(" z2:\n", z2.T)
print(" a2:\n", a2.T)
print(" ∂L/∂b2:\n", dL_db2)
print(" ∂L/∂W2:\n", dL_dW2.T)

print("Layer 1 gradients:")
print(" z1:\n", z1.T)
print(" a1:\n", a1.T)
print(" ∂L/∂b1:\n", dL_db1)
print(" ∂L/∂W1:\n", dL_dW1)
