import numpy as np

# Define the word embeddings for the vocabulary
# Using an example vocabulary of size 22 with 4-dimensional embeddings
vocabulary_size = 22
embedding_size = 4
word_embeddings = {
    "a": [0, 2, -1, 0],
    "an": [-1, 0, 0, 1],
    "the": [0, 2, -1, 0],
    "how": [1, 0, 1, 0],
    "why": [0, 0, 2, 1],
    "who": [1, 0, 0, 0],
    "what": [0, -1, 0, 1],
    "are": [0, 1, 1, 0],
    "is": [-1, 0, 0, 1],
    "am": [1, 0, 1, 0],
    "be": [0, 0, -1, 1],
    "was": [0, 2, 0, -2],
    "you": [0, 1, 0, 0],
    "we": [3, 0, -1, 0],
    "I": [1, 2, 0, 0],
    "they": [0, 0, 3, 1],
    "she": [-1, 2, 0, 0],
    "he": [0, 0, 0, 1],
    "she": [0, 0, -1, 0],
    "me": [0, 2, 0, 1],
    "him": [-1, 0, 2, 0],
    "her": [0, 0, -1, 1],
}


# Define the encoder with the specified weights and biases
def simple_encoder(word_embeddings):
    weights = np.array([[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, -1, 0, 0]]).T
    biases = np.array([0, 0, -1, 0])
    encoded = np.dot(word_embeddings, weights) + biases
    encoded = np.maximum(encoded, 0)  # Apply ReLU activation
    return encoded


# Define mean pooling function
def mean_pooling(encoded_sequence):
    # Calculate the sum across the sequence (columns) and divide by 3
    mean_pooled = np.mean(encoded_sequence, axis=0)
    return mean_pooled


# Define projection matrix for indexing (dimension reduction)
projection_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]]).T


# Process a sentence through the steps
def process_sentence(sentence, word_embeddings, encoder, pooling, projection):
    # Tokenize the sentence and get the corresponding word embeddings
    tokens = sentence.split()
    token_embeddings = np.array([word_embeddings[token] for token in tokens])

    # Encoding
    encoded_tokens = encoder(token_embeddings)

    # Mean Pooling
    text_embedding = pooling(encoded_tokens)

    # Indexing (dimensionality reduction)
    index_vector = np.dot(text_embedding, projection)

    return index_vector


# Vector storage for database vectors
vector_storage = []

# Example sentences
sentences = ["how are you", "who are you", "who am I"]

# Process each sentence to create the vector database
for sentence in sentences:
    vector_storage.append(
        process_sentence(
            sentence, word_embeddings, simple_encoder, mean_pooling, projection_matrix
        )
    )

# Query processing
query_sentence = "am I you"
query_vector = process_sentence(
    query_sentence, word_embeddings, simple_encoder, mean_pooling, projection_matrix
)

# Dot products (similarity estimation)
dot_products = np.dot(vector_storage, query_vector)

# Nearest Neighbor (retrieval)
nearest_neighbor_index = np.argmax(dot_products)
nearest_neighbor_sentence = sentences[nearest_neighbor_index]

# Print the results
print("Vector Database for sentences:\n", sentences)
print("Query sentence:\n", query_sentence)
print("Nearest Neighbor Sentence:\n", nearest_neighbor_sentence)
print("Dot products:\n", dot_products)
print("Nearest Neighbor Index:\n", nearest_neighbor_index)
