# Deep Learning and Neural Network Design

## Exercise 1

### [Autoencoder](https://lnkd.in/g2rM9iV2)

In this exercise, we explore the concept of an autoencoder, a type of neural network that learns to encode and decode data. The exercise provides a hands-on understanding of the architecture and operations of an autoencoder, including the encoding and decoding processes, as well as the loss function used to train the network.

- Encoder Architecture
  - Linear transformation from 4D to 3D.
  - ReLU activation.
  - Linear transformation from 3D to 2D (bottleneck).
  - ReLU activation.

- Decoder Architecture
  - Linear transformation from 2D to 3D.
  - ReLU activation.
  - Linear transformation from 3D to 4D.

1. Encoder Process: Apply linear transformation and ReLU to inputs, creating a bottleneck with fewer feature dimensions.

2. Decoder Process: Reconstruct original input dimensions from the bottleneck features using linear transformations and ReLU.

3. Loss Gradients and Backpropagation: Calculate MSE loss gradients as `2 * (Y - Y')` to initiate weight and bias updates.

Code: [autoencoder.py](./autoencoder.py)

## Exercise 2

### [Recurrent Neural Network (RNN)](https://lnkd.in/gDANw4iH)

This exercise dives into the fundamental mechanics of Recurrent Neural Networks (RNNs) by manually calculating the operations involved in a simple RNN. The purpose is to provide an intuitive understanding of how RNNs process sequences one step at a time, setting a foundation for more complex architectures like vision transformers (ViT).

1. Initialize hidden states to `[0, 0]`.
2. Combine the first input `(x1)` and hidden states linearly using weights `A` and `B`, apply ReLU activation to compute the new hidden states.
3. Use weights `C` to linearly combine hidden states to obtain the output `(y1)`.
4. Repeat steps `1-3` for `x2, x3, x4`, showcasing the sequential processing of inputs.

> Parameters: Utilize the same set of parameter matrices `(A, B, C)` for each input token, highlighting the recurrent nature of the network.

> Sequential Processing: Demonstrates the RNN's ability to process each input token sequentially, contrasting with models like Transformers that process all tokens in parallel.

Code: [recurrent_neural_network.py](./recurrent_neural_network.py)

## Exercise 3

### [Understanding the Transformer Model](https://lnkd.in/g39jcD7j)

This exercise demystifies the Transformer model, focusing on its core components: attention mechanisms and feed-forward networks.

1. Given: Input features from the previous block (5 positions).
2. Attention: Use a query-key attention module (QK) to calculate attention weights and obtain an attention weight matrix (A).
3. Attention Weighting: Apply the attention weight matrix to the input features, combining features across positions to produce attention-weighted features (Z).
4. FFN: First Layer: Increase the dimensionality of each feature from 3 to 4, applying weights and biases.
5. ReLU: Apply ReLU to set negative values to zero.
6. FFN: Second Layer: Reduce the dimensionality back from 4 to 3, preparing the output for the next block.

> Attention Mechanism: The attention mechanism is the heart of the Transformer, allowing the model to focus on different parts of the input sequence.

> The Feed-Forward Network processes each position's features, adjusting dimensions and applying non-linear transformations.

Code: [transformer.py](./transformer.py)

## Exercise 4

### [Generative Adversarial Network (GAN)](https://lnkd.in/gyKzNGDy)

In this exercise, we explore the workings of a Generative Adversarial Network (GAN) by manually calculating the operations within both the generator and discriminator components. This exercise provides a fundamental understanding of GAN architecture and the interaction between its components to generate new data that mimics the distribution of real data.

- Generator Architecture:
  1. First Layer: Linear transformation from 2D noise to an intermediate feature space (3D).
  2. ReLU Activation: Ensures non-linearity by setting negative values to zero.
  3. Second Layer: Linear transformation from the intermediate feature space (3D) to the generated data space (4D).
  4. ReLU Activation: Applied again to ensure non-linearity and positiveness in the generated data.

- Discriminator Architecture:
  1. First Layer: Linear transformation from the data space (4D) to an intermediate feature space (3D).
  2. ReLU Activation: Ensures non-linearity by setting negative values to zero.
  3. Second Layer: Linear transformation from the intermediate feature space (3D) to a single output (probability).
  4. Sigmoid Activation: Converts the output to a probability value, indicating the likelihood of the input being real data.

- Training Process:
  1. The generator creates "fake" data from random noise.
  2. The discriminator evaluates both real data and fake data from the generator.
  3. Loss functions are calculated for both the generator and discriminator to update their weights, aiming to improve the generator's ability to create realistic data and the discriminator's ability to distinguish real from fake.

- Objective:
  The ultimate goal is to train the generator to produce data indistinguishable from real data, as judged by the discriminator.

- Key Concepts:
  - Generative Models: Learn to generate new data similar to the training set.
  - Adversarial Training: A competitive process where the generator and discriminator improve through competition.

Code: [generative_adversarial_network.py](./generative_adversarial_network.py)

## Exercise 5

### [Switch Transformer and Sparse Mixture of Experts](https://www.linkedin.com/posts/tom-yeh_gemini-transformer-deeplearning-activity-7167154366821380096-e2YQ)

This exercise explores the Switch Transformer model, incorporating a Sparse Mixture of Experts (MoE) to handle large-scale models efficiently. The Switch Transformer extends traditional transformer models by routing input features to the most relevant expert, reducing computational load and allowing the model to scale to trillions of parameters.

1. Given: Input features from the previous block `(X1-X5)`.
2. Attention Pooling: Use an attention weight matrix `(A)` to pool features across positions, producing attention-weighted features `(Z1-Z5)`.
3. Calculate Gate Values: Multiply pooled features by a switch matrix to calculate gate values for each expert.
4. Route to Top Expert: Based on gate values, route each feature to the top expert, considering each expert's capacity.
5. Process by Experts: Each expert processes routed features with a linear layer and bias, combining features vertically.
6. Handle Capacity: If an expert's capacity is exceeded, the feature is passed through as is to the next block.

- Key Concepts:
  - Sparse Mixture of Experts (MoE): Allows the model to dynamically route inputs to the most relevant expert based on learned gate values.
  - Efficiency and Scalability: By utilizing a sparse gating mechanism, the Switch Transformer can scale efficiently to handle trillions of parameters, surpassing the capabilities of traditional dense models.
  - Expert Processing: Each expert specializes in processing certain types of input features, enhancing the model's overall performance and efficiency.

Code: [switch_transformer.py](./switch_transformer.py)
