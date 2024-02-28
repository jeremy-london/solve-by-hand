import numpy as np


# Function to find the matching word based on input embedding
def find_matching_word(input_embedding, vocab):
    # Compute the cumulative probability
    cumulative_prob = 0
    vocab_with_cumul = []
    for word, prob in reversed(vocab):
        cumulative_prob += prob
        vocab_with_cumul.insert(0, (word, np.round(cumulative_prob, 2)))

    print("vocab_ with cumulative probabilities:\n", vocab_with_cumul)

    # Find the word in the vocabulary where the input embedding falls within the cumulative probability
    closest_word = None
    for word, cum_prob in vocab_with_cumul:
        if input_embedding <= cum_prob:
            closest_word = word

    # if closest_word is not None:
    print(f"Closest Word for {input_embedding}:\n", closest_word)
    return closest_word


# Define vocabularies with probability distributions
vocab_1 = [
    ["I", 0.01],
    ["you", 0.01],
    ["they", 0.01],
    ["are", 0.01],
    ["am", 0.01],
    ["how", 0.50],
    ["why", 0.10],
    ["where", 0.10],
    ["who", 0.15],
    ["what", 0.10],
]
vocab_2 = [
    ["I", 0.01],
    ["you", 0.01],
    ["they", 0.01],
    ["are", 0.40],
    ["am", 0.40],
    ["how", 0.05],
    ["why", 0.05],
    ["where", 0.05],
    ["who", 0.01],
    ["what", 0.01],
]
vocab_3 = [
    ["I", 0.03],
    ["you", 0.50],
    ["they", 0.40],
    ["are", 0.01],
    ["am", 0.01],
    ["how", 0.01],
    ["why", 0.01],
    ["where", 0.01],
    ["who", 0.01],
    ["what", 0.01],
]

# Input embeddings
input_embeddings = [0.34, 0.52, 0.92, 0.65]

# Sample words from vocabularies based on input embeddings
words = [
    find_matching_word(input_embeddings[3], vocab_1),
    find_matching_word(input_embeddings[2], vocab_2),
    find_matching_word(input_embeddings[1], vocab_3),
]


# Construct the sentence
sampled_sentence = " ".join(words)
print("Sampled Sentence:\n", sampled_sentence)
