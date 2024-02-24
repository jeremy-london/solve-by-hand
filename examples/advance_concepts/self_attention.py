import numpy as np

# Define a set of 4 feature vectors (6-D) with the given values
features = np.array(
    [
        [2, 0, 0, 2],  # x1
        [0, 1, 0, 0],  # x2
        [0, 2, 1, 0],  # x3
        [0, 0, 1, 1],  # x4
        [2, 0, 0, 0],  # x5
        [1, 0, 1, 1],  # x6
    ]
)

# Define linear transformation matrices WQ and WK with the given values
WQ = np.array(
    [
        [1, 1, 0, 0, 0, 0],  # WQ row 1
        [0, 1, 0, 1, 0, 0],  # WQ row 2
        [0, 0, 1, 0, 1, 1],  # WQ row 3
    ]
)

WK = np.array(
    [
        [0, 0, 1, 0, 0, 0],  # WK row 1
        [0, 1, 0, 0, 0, 0],  # WK row 2
        [1, 0, 0, 0, 0, -1],  # WK row 3
    ]
)

# WV is not specified so it will be considered as an identity for this example
WV = np.array(
    [
        [10, 0, 0, 0, 0, 0],  # WK row 1
        [0, 0, 0, 10, 0, 0],  # WK row 2
        [0, 10, 0, 0, 0, 0],  # WK row 3
    ]
)

# Calculate Q and K
Q = np.matmul(WQ, features)  # Queries
K = np.matmul(WK, features)  # Keys

# MatMul: Multiply K^T and Q
attention_scores = np.matmul(K.T, Q)

# Printing the intermediate results
print("Query Vectors (Q):\n", Q)
print("Key Vectors Transposed (K^T):\n", K.T)
print("Attention Scores (K^T * Q):\n", attention_scores)

# Define dk
dk = 3

# Custom scaling function
scaled_attention_scores = np.where(
    attention_scores < 0,
    np.floor(attention_scores / np.sqrt(dk)) + 1,  # Adjust negative values more gently
    np.floor(attention_scores / 2),  # Apply original scaling for non-negative values
)

print("Adjusted Scaled Attention Scores:\n", scaled_attention_scores)

# Apply the approximation e^x ≈ 3^x
approx_exp_attention_scores = np.where(
    scaled_attention_scores < 0, 0, 3**scaled_attention_scores
)

print("Approximated Exponential Attention Scores:\n", approx_exp_attention_scores)

# Softmax: Sum across each column
column_sums = np.sum(approx_exp_attention_scores, axis=0)
print("Column Sums:\n", column_sums)

# Softmax: 1 / sum
# For each column, divide each element by the column sum
attention_weight_matrix_A = np.round((approx_exp_attention_scores / column_sums), 1)

print("Attention Weight Matrix (A):\n", attention_weight_matrix_A)

# Calculate Value Vectors (V)
V = np.matmul(WV, features)  # Values
print("Value Vectors (V):\n", V)

# Perform the matrix multiplication to get Z, the attention weighted features
Z = np.dot(V, attention_weight_matrix_A.T)

print("Attention Weighted Features (Z):\n", Z)
