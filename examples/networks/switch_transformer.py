import numpy as np

# Step 1: Define inputs and attention matrix
X = np.array([[5, 0, 1], [6, 2, 0], [0, 4, 1], [7, 0, 1], [0, 3, 0]])
A = np.array(
    [
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]
)

# Step 2: Attention Pooling
Z = A.T.dot(X)
print("Attention pooled features:\n", Z.T)

# Step 3: Calculate gate values and route to the top expert
switch_matrix = np.array([[1, -1, 0], [1, 0, -2], [0, 1, 1]])
gate_values = np.dot(Z, switch_matrix.T)
print("Gate values:\n", gate_values.T)
top_experts = np.argmax(gate_values, axis=1)
print("Top experts:\n", [chr(65 + i) for i in top_experts])  # Expert IDs A, B, C

# Expert definitions
experts = [
    np.array([[1, 0, -1], [0, 1, 0], [0, 0, -1]]),  # Expert A
    np.array([[0, 0, -1], [1, 0, 0], [1, 1, -1]]),  # Expert B
    np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]),  # Expert C
]

biases = [
    np.array([0, 0, 1]),  # Bias for Expert A
    np.array([0, 0, 1]),  # Bias for Expert B
    np.array([0, 1, 0]),  # Bias for Expert C
]

# Step 4: Process features by experts
output = np.zeros_like(X)
capacity = {0: 2, 1: 2, 2: 2}  # Capacity for each expert

# Process features by experts based on the top expert identification and capacity
for i, expert_id in enumerate(top_experts):
    if capacity[expert_id] > 0:
        output[i] = np.dot(experts[expert_id], Z[i]) + biases[expert_id]
        capacity[expert_id] -= 1
    else:
        # If expert's capacity is exceeded, pass through Z as is
        output[i] = Z[i]

    print(
        f"Processing Z{i+1} with Expert {chr(65 + expert_id)}:\n", output[i]
    )  # Expert IDs A, B, C

print("Output features:\n", output.T)
