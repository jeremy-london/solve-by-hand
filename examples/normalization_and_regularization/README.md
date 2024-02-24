# Normalization and Regularization

This folder contains exercises and examples related to normalization and regularization techniques in deep learning models.

## Exercise 1

### [Batch Normalization](https://lnkd.in/gVjknYkU)

Batch normalization is a technique used to normalize the inputs of each layer in a neural network. By doing so, it reduces the internal covariate shift and allows for smoother and faster training. In this exercise, we'll walk through the steps of batch normalization applied to a mini-batch of training examples.

1. Given
   - A mini-batch of `4` training examples, each with `3` features.

2. Linear Layer
   - Multiply the input features with the weights and add biases to obtain new features.

3. ReLU Activation
   - Apply the Rectified Linear Unit (ReLU) activation function to introduce non-linearity. Negative values are set to zero.

4. Batch Statistics
   - Compute the sum, mean, variance, and standard deviation across the four examples in this mini-batch for each feature dimension.

5. Shift to `Mean = 0`
   - Subtract the mean from the activation values for each training example.
   - The goal is for the four activation values in each dimension to average to zero.

6. Scale to `Variance = 1`
   - Divide by the standard deviation for each feature dimension.
   - The goal is for the four activation values in each dimension to have a variance equal to one.

7. Scale & Shift
   - Multiply the normalized features by a linear transformation matrix and add a bias term.
   - The diagonal elements and the last column in the transformation matrix are trainable parameters learned by the network.
   - The intention is to scale and shift the normalized feature values to new means and variances, which are learned by the network.

Code: [batch_normalization.py](./batch_normalization.py)

## Exercise 2

### Dropout and MSE Loss Gradient Adjustment

Dropout is a straightforward yet effective technique to reduce overfitting and enhance generalization in neural networks. This exercise provides a practical experience in implementing dropout and utilizing Mean Square Error (MSE) loss gradients for weight adjustment.

1. Given:
   - A training set comprising 2 examples, `X1` and `X2`.

2. Random Dropout Implementation: Generate random numbers to decide which neurons to keep or drop based on predefined probabilities (`p > 0.5` for the first dropout layer, `p > 0.33` for the second).

3. Dropout Matrices:
   - Calculate scaling factors `(1 / (1-p))` to adjust the activation of neurons that are kept.
   - Apply these factors to create dropout matrices, effectively dropping certain neurons while scaling the activation of others.

4. Feed Forward with Dropout: Perform the feed forward process, applying ReLU activations and dropout according to the dropout matrices.

5. MSE Loss Gradients: Calculate the MSE loss gradients as `2 * (Y - Y')`, where `Y` is the network output and `Y'` are the targets. This step is crucial for understanding how to adjust weights to minimize loss.

6. Weight Adjustment: Adjust the original weights based on the loss gradients, demonstrating the backpropagation process without manually setting new target weights.

7. Inference without Dropout:
   - For inference, deactivate dropout by using identity matrices as dropout masks, ensuring all neurons contribute to the process.
   - Perform a forward pass to make predictions on unseen data, showcasing the network's ability to generalize.

Code: [dropout.py](./dropout.py)
