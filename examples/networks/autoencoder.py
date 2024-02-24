import numpy as np


def linear_transform(X, weights, bias):
    return np.dot(X, weights.T) + bias


def relu(X):
    return np.maximum(0, X)


def forward_pass(
    X,
    encoder_weights_1,
    encoder_bias_1,
    encoder_weights_2,
    encoder_bias_2,
    decoder_weights_1,
    decoder_bias_1,
    decoder_weights_2,
    decoder_bias_2,
):
    # Encoder
    hidden_1 = relu(linear_transform(X, encoder_weights_1, encoder_bias_1))
    print("Hidden Encoder:\n", hidden_1.T)
    bottleneck = relu(linear_transform(hidden_1, encoder_weights_2, encoder_bias_2))
    print("Encoder Bottleneck:\n", bottleneck.T)

    # Decoder
    hidden_2 = relu(linear_transform(bottleneck, decoder_weights_1, decoder_bias_1))
    print("Hidden Decoder:\n", hidden_2.T)
    reconstructed = linear_transform(hidden_2, decoder_weights_2, decoder_bias_2)
    print("Reconstructed Outputs for Y:\n", reconstructed.T)

    return reconstructed


# Calculate the loss gradients for backpropagation
def calculate_gradients(Y, targets):
    return 2 * (Y - targets)


# Define the training examples
X = np.array([[1, 2, 3, 1], [1, 1, 2, 1], [2, 2, 4, 2], [1, 0, 1, 1]])

# Define weights and biases for the encoder
encoder_weights_1 = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [-1, 0, 1, 0]])
encoder_bias_1 = np.array([0, 0, -1])
encoder_weights_2 = np.array([[1, 0, 1], [-1, 1, 0]])
encoder_bias_2 = np.array([0, 0])

# Define weights and biases for the decoder
decoder_weights_1 = np.array([[1, 0], [0, 1], [1, -1]])
decoder_bias_1 = np.array([0, 1, 0])
decoder_weights_2 = np.array([[1, 0, -1], [1, -1, 0], [0, 0, 1], [0, 1, 1]])
decoder_bias_2 = np.array([0, 0, 1, -3])

# Forward pass to get the reconstructed outputs
Y = forward_pass(
    X,
    encoder_weights_1,
    encoder_bias_1,
    encoder_weights_2,
    encoder_bias_2,
    decoder_weights_1,
    decoder_bias_1,
    decoder_weights_2,
    decoder_bias_2,
)

# Targets are the same as inputs for an autoencoder
Y_prime = X
gradients = calculate_gradients(Y, Y_prime)

print("Gradients for Backpropagation:\n", gradients.T)
