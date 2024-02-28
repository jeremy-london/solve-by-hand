# Advanced Concepts in Deep Learning

## Exercise 1

### [Self Attention](https://lnkd.in/gDW8Um4W)

In this advanced exercise, we explore the self-attention mechanism, a key component of the Transformer model. Self-attention allows the model to weigh the importance of different words in a sentence when making predictions. This mechanism is particularly effective for capturing long-range dependencies in sequences, making it a popular choice for natural language processing tasks.

1. Given Features
   - We begin with a set of 4 feature vectors, each consisting of 6 dimensions.

2. Query, Key, and Value Calculation
   - Compute query vectors `(q1, q2, q3, q4)`, key vectors `(k1, k2, k3, k4)`, and value vectors `(v1, v2, v3, v4)` by multiplying the feature vectors with linear transformation matrices `WQ`, `WK`, and `WV`, respectively.
   - Note: The term "self" implies that both queries and keys are derived from the same set of features.

3. Preparing for Matrix Multiplication
   - Duplicate the query vectors.
   - Transpose the key vectors.

4. Matrix Multiplication
   - Perform a matrix multiplication between the transposed key vectors and the query vectors, resulting in the dot product between each pair of query and key vectors.

5. Scaling
   - Scale each element of the resulting matrix by the square root of dk (dimension of key vectors), ensuring normalization of the impact of key vector dimensions on matching scores.

6. Softmax Transformation: Exponential
   - Apply the softmax transformation by exponentiating each element in the scaled matrix.
   - For simplicity, approximate `e^x` with `3^x`.

7. Softmax Transformation: Summation
   - Sum the elements across each column of the exponential transformation result.

8. Softmax Transformation: Normalization
   - Normalize each column of the exponential matrix by dividing each element by the sum of the respective column, ensuring each column sums to 1.

9. Matrix Multiplication with Values
   - Multiply the value vectors with the resulting attention weight matrix, yielding attention-weighted features.
   - These features are then fed into the position-wise feedforward network in the subsequent layer.

Code: [self_attention.py](./self_attention.py)

## Exercise 2

### [Vector Database and Similarity Search](https://lnkd.in/gTanDTMj)

In this advanced exercise, we construct a simple vector database to understand the mechanics behind similarity searches in natural language processing. Utilizing word embeddings and a neural encoder, we convert sentences into vector representations. These vectors can then be compared using dot product to find the most similar entries in our database.

Key concepts covered:

- Word Embeddings: Initialize a fixed-size vector for each unique word.
- Encoding: Apply a linear transformation followed by a ReLU activation to word embeddings.
- Mean Pooling: Aggregate encoded word vectors by averaging to create a sentence-level embedding.
- Projection: Reduce dimensions of the pooled vector for efficient storage and retrieval.
- Similarity Search: Use dot product to find the closest vector in the database to a query vector.

Example workflow:

1. Input sentence "how are you" is split into tokens `["how", "are", "you"]`.
2. Tokens are converted to embeddings and encoded individually.
3. Mean pooling is applied to the encoded tokens to obtain a single vector.
4. The vector is projected to a lower-dimensional space.
5. The resulting vector is stored in a database.
6. For a query sentence, the same process is followed to obtain a query vector.
7. The query vector is compared to all vectors in the database using dot product.

The outcome of this process is a foundational technique used in various applications such as chatbots, search engines, and recommendation systems.

Code: [vector_database.py](./vector_database.py)

## Exercise 3

### [Mixture of Experts (MoE)](https://lnkd.in/gPFdQdsW)

In this advanced exercise, we explore the Mixture of Experts (MoE) model, a powerful architecture for handling complex data distributions. MoE is particularly effective for tasks that require a model to learn from a diverse set of data, such as language translation, image recognition, and recommendation systems.

1. Inputs: The mixture of experts (MoE) model takes in two tokens distinguished by colors.
2. Gate Network Decision for `X1`: After processing the first token, the gate network assigns the highest weight to `Expert 2`, indicating its selection for further processing.
3. Expert 2 Processing for `X1`: `Expert 2` processes the first token.
4. Gate Network Decision for `X2`: For the second token, the gate network assigns the highest weight to `Expert 1`, indicating that it should take over the processing.
5. `Expert 1` Processing for `X2`: `Expert 1` processes the second token.
6. Final Output with ReLU: The outputs from the experts for each token are then passed through a ReLU activation function to produce the final processed outputs.

Code: [mixture_of_experts.py](./mixture_of_experts.py)

## Exercise 4: Mamba S6 - Dynamic Sequence Modeling

Mamba S6 introduces a novel approach to sequence modeling, advancing beyond traditional RNNs and Transformer architectures by leveraging a dynamic, structured state-space model. Developed by Albert Gu and Tri Dao, Mamba excels in processing complex sequences across diverse fields such as language processing, genomics, and audio analysis, showcasing exceptional efficiency and performance.

### Overview of Mamba S6

1. Dynamic Weight Application: Each step in the input sequence utilizes a unique set of weight parameters (A, B, C), enabling flexible adaptation to the sequence's context. This contrasts with conventional RNNs that typically reuse the same weights across all steps.

2. Linear Processing without Non-linearity: Mamba S6 processes sequences through linear transformations, omitting non-linear activation functions. This design choice simplifies the computational model while still capturing the essential dynamics of the input sequence.

3. Structured Matrices for State Transformation: The model uses structured matrices to update hidden states, starting with a specific structure that is dynamically adjusted in later operations. This allows for selective emphasis on certain aspects of the input data, enhancing the model's ability to process complex sequences.

4. Selective State Spaces for Efficient Computation: Mamba's architecture, inspired by FlashAttention, incorporates selective state space layers that optimize computation and memory usage, particularly for long sequences.

5. Superior Performance and Scalability: By addressing the computational challenges inherent in processing long sequences, Mamba achieves faster inference times and linear scaling with sequence length, outperforming Transformer-based models in efficiency and throughput.

### Mamba vs Transformers

Transformers revolutionized sequence modeling with their attention mechanisms, enabling parallel processing of sequences and capturing complex relationships within the data. However, they struggle with computational efficiency for long sequences. Mamba S6 addresses these limitations by using selective state spaces and structured transformations, offering a more computationally efficient model for long sequences without compromising on performance.

### Unique Contributions of Mamba S6

- Selective State-Space Modeling: Filters irrelevant information, focusing on relevant data for efficient sequence processing.
- Hardware-aware Design: Optimizes computation for modern hardware, particularly GPUs, enhancing computational speed and reducing memory requirements.
- Simplified Architecture: Eliminates traditional attention and MLP blocks, resulting in a lighter and faster model that scales linearly with sequence length.

Mamba S6 sets a new standard for sequence modeling, combining the efficiency of linear-time processing with the flexibility of dynamic weight selection, making it an innovative tool for a wide range of applications in AI and machine learning.

Code: [mamba_s6.py](./mamba_s6.py)

## Exercise 5: LLM Sampling

In this exercise, we delve into the mechanics of sampling in Large Language Models (LLMs) using a hands-on approach to understand how each word in a sentence is drawn from a probability distribution.

1. Input Embeddings: Begin with a sequence of input embeddings, each represented as a column vector.
2. LLM Prediction: These embeddings are fed into an LLM, which predicts the probability distribution of the next word over a vocabulary.
3. Cumulative Distribution: Manually calculate the cumulative distribution, starting at zero and adding each probability to the total as you move up the list up to 100.
4. Random Sampling: Draw a random number from the embedded input `(e.g., 0.65)` and find where it falls within the cumulative distribution within the vocabulary representation.
5. Word Selection: Select the word corresponding to this range `(e.g., "how 0.5")`.
6. Repeat for Subsequent Words: Continue the process for the next output words, drawing new random numbers and selecting the appropriate words.

Code: [llm_sampling.py](./llm_sampling.py)
