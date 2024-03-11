import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_reward(embeddings, weights, bias, output_weights, output_bias):
    feature_vectors = np.dot(weights, embeddings) + bias[:, np.newaxis]
    activated_vectors = relu(feature_vectors)
    print("Activated vectors:\n", activated_vectors)

    mean_pooled_vector = np.mean(activated_vectors, axis=1)
    print("Mean pooled vector:\n", mean_pooled_vector)

    # Apply output layer weights and bias
    reward_output_vector = np.dot(output_weights, mean_pooled_vector) + output_bias

    # Sum up the output and multiply by 2 as per your original code's logic
    final_reward = 2 * reward_output_vector.sum()
    return final_reward


# RM weights and bias
reward_model_weights = np.array([[1, 0, 1], [0, 1, 0], [1, 0, -1], [1, 1, 0]])
reward_model_bias = np.array([0, 0, 0, 0])
reward_output_layer = np.array([3, 3, 3, -3])
reward_output_bias = -1

# Word embeddings dictionary
word_embeddings = {
    "him": np.array([0, 1, 0]),
    "her": np.array([1, 0, 1]),
    "them": np.array([1, 0, 0]),
    "is": np.array([1, 1, 1]),
    "doc": np.array([1, 1, 0]),
    "CEO": np.array([0, 1, 1]),
    "[S]": np.array([-1, -1, -1]),
}

# Input prompts
input_prompts = {"loser": ["doc", "is", "him"], "winner": ["doc", "is", "them"]}

# Calculate rewards for each prompt
rewards = {}
for prompt_type in input_prompts:
    prompt = input_prompts[prompt_type]
    embeddings = np.array([word_embeddings[word] for word in prompt]).T
    print(f"Embeddings for {prompt_type} prompt:\n", embeddings)
    rewards[prompt_type] = get_reward(
        embeddings,
        reward_model_weights,
        reward_model_bias,
        reward_output_layer,
        reward_output_bias,
    )

print("Winner reward:\n", rewards["winner"])
print("Loser reward:\n", rewards["loser"])

# Calculate the reward gap
reward_gap = rewards["winner"] - rewards["loser"]
print("Reward gap:\n", reward_gap)

# Map reward gap to a probability value as prediction
prediction = sigmoid(reward_gap)
print("Predicted σ:\n", prediction)

# Calculate loss gradient
target = 1
loss_gradient = round(prediction - target, 1)

# Define the prompt
llm_prompt = ["[S]", "CEO", "is"]
llm_attention_matrix = np.array([[5, 1, 0], [0, 4, 1], [0, 0, 4]])
llm_feed_forward_weights = np.array([[1, 0, -1], [-1, 1, 0]])
llm_feed_forward_bias = np.array([2, 0])

# Output probability layer
llm_output_probabilities = {
    "him": {"weights": np.array([1, 1]), "bias": 0},
    "her": {"weights": np.array([0, 1]), "bias": 0},
    "them": {"weights": np.array([1, 0]), "bias": 0},
    "is/are": {"weights": np.array([-1, 1]), "bias": 1},
    "doc": {"weights": np.array([2, 0]), "bias": -2},
    "CEO": {"weights": np.array([2, 0]), "bias": -1},
}

llm_prompt_embeddings = np.array([word_embeddings[word] for word in llm_prompt]).T
print("Prompt embeddings:\n", llm_prompt_embeddings)

# Apply attention to the prompt embeddings
llm_feature_vectors = np.dot(llm_prompt_embeddings, llm_attention_matrix)
print("Feature vectors after applying attention:\n", llm_feature_vectors)

llm_transformed_feature_vectors = relu(
    np.dot(llm_feed_forward_weights, llm_feature_vectors)
    + llm_feed_forward_bias.reshape(-1, 1)
)
print("Transformed feature vectors:\n", llm_transformed_feature_vectors)

llm_output_layer_weights = {
    "him": np.array([1, 1]),
    "her": np.array([0, 1]),
    "them": np.array([1, 0]),
    "is": np.array([-1, 1]),
    "doc": np.array([2, 0]),
    "CEO": np.array([2, 0]),
}

llm_output_layer_biases = {
    "him": 0,
    "her": 0,
    "them": 0,
    "is": 1,
    "doc": -2,
    "CEO": -1,
}

# Apply a linear layer to map each transformed feature vector to a probability distribution over the vocabulary
output_probabilities = {}
for word, weights in llm_output_layer_weights.items():
    bias = llm_output_layer_biases[word]
    output_probabilities[word] = np.dot(weights, llm_transformed_feature_vectors) + bias

# Print the output probabilities for each word
print("Output probabilities for each word:")
for word, probabilities in output_probabilities.items():
    print(f"{word}: {probabilities}")

# Find the word with the highest probability for each column (position)
highest_prob_words = [
    max(output_probabilities, key=lambda word: output_probabilities[word][i])
    for i in range(3)
]

print("Highest probability words:\n", highest_prob_words)

# Map the highest probability words to their original embeddings
new_array_from_embeddings = np.array(
    [word_embeddings[word] for word in highest_prob_words]
).T
print("Highest probability words embeddings:\n", new_array_from_embeddings)

# Call the function to calculate the reward with new embeddings
llm_output_reward = get_reward(
    new_array_from_embeddings,
    reward_model_weights,
    reward_model_bias,
    reward_output_layer,
    reward_output_bias,
)
print(f"LLM Output reward: {llm_output_reward}")

# Loss is the negative of the final reward
loss = -llm_output_reward.sum()
print(f"Loss: {loss}")

# Loss gradient is set to -1
loss_gradient = -1
print(f"Loss gradient: {loss_gradient}")
