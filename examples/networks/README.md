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

Code: [transformer_model.py](./transformer_model.py)
