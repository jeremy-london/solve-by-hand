import numpy as np

# Provided random sequence
random_sequence = np.array(
    [
        0.61,
        0.39,
        0.75,
        0.40,
        0.65,
        0.42,
        0.23,
        0.19,
        0.93,
        0.42,
        0.87,
        0.53,
        0.27,
        0.69,
        0.50,
        0.11,
        0.42,
    ]
)

# Training and unseen input features
X_train = np.array([[3, 4], [5, 1]])
X_unseen = np.array([[3, 2], [3, 1]])
Y_prime = np.array([[-4, 7], [10, 5]])

# Weights and biases for each layer
weights_1 = np.array([[1, 0], [1, 1], [0, 1], [1, -1]])
bias_1 = np.array([0, 0, 1, 0])

weights_2 = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, -1, -1]])
bias_2 = np.array([0, 0, 1])

weights_3 = np.array([[1, -1, 0], [0, 1, -1]])
bias_3 = np.array([0, -2])


# Activation function
def relu(x):
    return np.maximum(0, x)


# Generate dropout masks from random sequence
def generate_dropout_mask_from_sequence(sequence, p, size):
    scale_factor = 1 / (1 - p)  # Calculate the scale factor based on p
    rounded_scale_factor = np.round(scale_factor, 1)
    mask = np.zeros(size)  # Initialize the mask with zeros
    for i, val in enumerate(sequence[:size]):
        mask[i] = (
            rounded_scale_factor if val > p else 0
        )  # Apply scale factor if val > p, else drop
    return np.diag(mask)


# Dropout masks
p1 = 0.5  # Dropout probability for the first layer
dropout_mask_1 = generate_dropout_mask_from_sequence(
    random_sequence, p1, 4
)  # For the first dropout layer
print("Dropout Mask 1:\n", dropout_mask_1)

p2 = 0.33  # Dropout probability for the second layer
dropout_mask_2 = generate_dropout_mask_from_sequence(
    random_sequence[4:], p2, 3
)  # For the second dropout layer
print("Dropout Mask 2:\n", dropout_mask_2)


# Forward pass function
def forward_pass(
    X,
    dropout_mask_1,
    dropout_mask_2,
    weights_1,
    bias_1,
    weights_2,
    bias_2,
    weights_3,
    bias_3,
):
    # Layer 1
    Z1 = relu(np.dot(X, weights_1.T) + bias_1)
    print("Layer 1 Outputs:\n", Z1.T)

    Z1_after_dropout_mask_1 = np.dot(Z1, dropout_mask_1)
    print("Layer 1 Outputs after dropout:\n", Z1_after_dropout_mask_1.T)

    # Layer 2
    Z2 = relu(np.dot(Z1_after_dropout_mask_1, weights_2.T) + bias_2)
    print("Layer 2 Outputs:\n", Z2.T)
    Z2_after_dropout_mask_2 = np.dot(Z2, dropout_mask_2)
    print("Layer 2 Outputs after dropout:\n", Z2_after_dropout_mask_2.T)

    # Layer 3
    Y = np.dot(Z2_after_dropout_mask_2, weights_3.T) + bias_3

    return Y


# Training phase
Y_train = forward_pass(
    X_train,
    dropout_mask_1,
    dropout_mask_2,
    weights_1,
    bias_1,
    weights_2,
    bias_2,
    weights_3,
    bias_3,
)

gradients_initial = Y_train.T - Y_prime
gradients_final = gradients_initial * 2

print("Y prime Outputs:\n", Y_prime)
print("Y train Outputs:\n", Y_train)
print("gradients_initial:\n", gradients_initial)
print("gradients_final:\n", gradients_final)

print("\nInference for unseen data:\n")

# Set dropout masks to identity matrices for inference (as dropout is deactivated)
dropout_mask_1_inference = np.eye(4)  # Identity matrix for the first dropout layer
print("Inference Dropout Mask 1 Inference:\n", dropout_mask_1_inference)
dropout_mask_2_inference = np.eye(3)  # Identity matrix for the second dropout layer
print("Inference Dropout Mask 2 Inference:\n", dropout_mask_2_inference)

# Desired new weights and biases
weights_1_new_goal = np.array([[1, 0], [1, 1], [-1, 1], [1, -1]])
bias_1_new_goal = np.array([0, 1, 1, 0])
weights_2_new_goal = np.array([[1, 0, 1, 1], [1, 1, 1, 0], [1, 0, -1, 0]])
bias_2_new_goal = np.array(
    [
        0,
        0,
        1,
    ]
)
weights_3_new_goal = np.array([[1, 1, 0], [0, 1, -1]])
bias_3_new_goal = np.array([0, -1])

# Calculating "hypothetical gradients" for demonstration
# Hypothetical gradients are the difference between the goal and the original
hypothetical_gradients_w1 = weights_1_new_goal - weights_1
hypothetical_gradients_b1 = bias_1_new_goal - bias_1
hypothetical_gradients_w2 = weights_2_new_goal - weights_2
hypothetical_gradients_b2 = bias_2_new_goal - bias_2
hypothetical_gradients_w3 = weights_3_new_goal - weights_3
hypothetical_gradients_b3 = bias_3_new_goal - bias_3

# Apply updates to original weights to achieve new weights
weights_1_updated = weights_1 + hypothetical_gradients_w1
bias_1_updated = bias_1 + hypothetical_gradients_b1
weights_2_updated = weights_2 + hypothetical_gradients_w2
bias_2_updated = bias_2 + hypothetical_gradients_b2
weights_3_updated = weights_3 + hypothetical_gradients_w3
bias_3_updated = bias_3 + hypothetical_gradients_b3

# Verify the update (this should match the desired new weights and biases)
print("Updated Weights 1:\n", weights_1_updated)
print("Updated Bias 1:\n", bias_1_updated)
print("Updated Weights 2:\n", weights_2_updated)
print("Updated Bias 2:\n", bias_2_updated)
print("Updated Weights 3:\n", weights_3_updated)
print("Updated Bias 3:\n", bias_3_updated)

# Assuming X_unseen and forward_pass function are defined, perform inference
Y_inference = forward_pass(
    X_unseen,
    dropout_mask_1_inference,
    dropout_mask_2_inference,
    weights_1_updated,
    bias_1_updated,
    weights_2_updated,
    bias_2_updated,
    weights_3_updated,
    bias_3_updated,
)

print("Inference Outputs:\n", Y_inference.T)
