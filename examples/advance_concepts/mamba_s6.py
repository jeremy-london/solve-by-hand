import numpy as np


# Function to calculate new hidden states and output
def calculate_output_and_hidden_states(input_seq, layers, initial_hidden_states):
    outputs = []
    hidden_states = initial_hidden_states

    for i, input_val in enumerate(input_sequence):
        if i < len(layers):
            layer = layers[i]
        else:
            layer = layers[-1]

        weights = np.dot(layer, input_seq)
        print(f"Weights {i}:\n", weights)

        A, B, C = weights[:2], structured, weights[2:]
        print(f"A {i}:\n", A)
        print(f"B {i}:\n", B)
        print(f"C {i}:\n", C)

        # if i > 1 flip structured matrix
        if i > 1:
            B = B * -1
        # sequence = np.concatenate([[input_val], hidden_states])
        # print(f"Sequence {i}:\n", sequence)

        # Update hidden states
        hidden_states = np.dot(A, input_val) + np.dot(B, hidden_states)
        print(f"Updated Hidden States {i}:\n", hidden_states)

        # Calculate output
        output = np.dot(C, hidden_states)
        print(f"Output {i}:\n", output)

        outputs.append(output)
    return outputs


# Input sequence
input_sequence = np.array([3, 4, 5, 6])

# Structured matrix
structured = np.array([[1, 0], [0, -1]])

# Layer matrices
layers = [
    np.array([[1, -1, 0, 0], [0, -1, 0, 1], [1, 0, -1, 0], [1, 0, 0, -1]]),
    np.array([[1, 0, -1, 0], [0, 1, 0, -1], [1, -1, 0, 0], [0, 0, -1, 1]]),
    np.array([[-1, 0, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]]),
    np.array([[1, -1, 0, 0], [0, 0, -1, 1], [1, 0, 0, 0], [0, -1, 1, 0]]),
]

# Initial hidden states
hidden_states = np.array([0, 0])

# Calculate outputs
outputs = calculate_output_and_hidden_states(input_sequence, layers, hidden_states)

print(f"Output Sequence:\n", outputs)
