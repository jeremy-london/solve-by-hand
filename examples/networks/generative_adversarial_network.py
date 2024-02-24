import numpy as np


# Function to apply ReLU activation
def relu(x):
    return np.maximum(0, x)


# Generator forward pass
def generator_forward(
    N, gen_first_weights, gen_first_bias, gen_second_weights, gen_second_bias
):
    layer1 = relu(np.dot(N, gen_first_weights.T) + gen_first_bias)
    print("Generator Layer 1:\n", layer1.T)
    F = relu(np.dot(layer1, gen_second_weights.T) + gen_second_bias)
    return F


# Discriminator forward pass
def discriminator_forward(
    input_data,
    disc_first_weights,
    disc_first_bias,
    disc_second_weights,
    disc_second_bias,
):
    layer1 = relu(np.dot(input_data, disc_first_weights.T) + disc_first_bias)
    print("Discriminator Layer 1:\n", layer1.T)
    Z = relu(np.dot(layer1, disc_second_weights.T) + disc_second_bias)
    print("Z:\n", Z.T)
    Y = np.round(1 / (1 + np.exp(-Z)), 1)  # Sigmoid activation
    return Y


# Define noise vectors (N) and real data vectors (X)
N = np.array([[1, 1], [1, 0], [0, 1], [1, -1]])
X = np.array([[2, 1, 2, 1], [3, 1, 3, 1], [3, 1, 4, 1], [4, 1, 3, 1]])

# Generator first layer weights, biases, and ReLU activation
gen_first_weights = np.array([[1, 1], [0, 1], [-1, 1]])
gen_first_bias = np.array([0, 2, 0])
gen_second_weights = np.array([[-1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]])
gen_second_bias = np.array([0, 0, 0, 1])

# Discriminator first layer weights, biases, and ReLU activation
disc_first_weights = np.array([[1, 0, 0, -1], [0, 1, 1, 0], [0, 0, 1, -1]])
disc_first_bias = np.array([0, 0, 1])
disc_second_weights = np.array([[1, 1, -1]])
disc_second_bias = np.array([-1])

# Generating fake data
F = generator_forward(
    N, gen_first_weights, gen_first_bias, gen_second_weights, gen_second_bias
)
print("Generated Fake Data:\n", F.T)

# Training Discriminator
D_fake = discriminator_forward(
    F, disc_first_weights, disc_first_bias, disc_second_weights, disc_second_bias
)
print("Discriminator Output for Fake Data:\n", D_fake.T)
D_real = discriminator_forward(
    X, disc_first_weights, disc_first_bias, disc_second_weights, disc_second_bias
)
print("Discriminator Output for Real Data:\n", D_real.T)


# Compute mean loss gradients for Discriminator (fake data)
YD_fake = np.array([0, 0, 0, 0])
fake_loss_gradients_discriminator_mean = np.mean(D_fake - YD_fake, axis=1)
print(
    "Mean Loss Gradients Discriminator for Fake Data:\n",
    fake_loss_gradients_discriminator_mean,
)

# Compute mean loss gradients for Discriminator (real data)
YD_real = np.array([1, 1, 1, 1])
real_loss_gradients_discriminator_mean = np.mean(D_real - YD_real, axis=1)
print(
    "Mean Loss Gradients Discriminator for Real Data:\n",
    real_loss_gradients_discriminator_mean,
)

# Training Generator
# Compute mean loss gradients for Generator
YG = np.array([1, 1, 1, 1])
loss_gradients_generator_mean = np.mean(D_fake - YG, axis=1)
print("Mean Loss Gradients Generator:\n", loss_gradients_generator_mean)
