import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
import os
import json
import pickle

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.

    Parameters:
        filepath (str): Path to the CSV file containing the data.

    Returns:
        tuple: Preprocessed data including:
            - A list of texts (list of str).
            - Encoded labels (list of int).
            - A list of unique labels (list of str).
    """
    # Load the data from CSV
    data = pd.read_csv(filepath)
    
    # Drop rows with missing values
    data = data.dropna()
    # data=data.query('Person=="A"')
    
    # Encode categories into integers

    category_to_int = {
        'Greetings': 0,
        'Engagement': 1,
        'Account Verification': 2,
        'Problem Investigation': 3,
        'Problem Resolution': 4,
        'Closure': 5
    }
    
    # Save to JSON file
    with open('category_to_int.json', 'w') as f:
        json.dump(category_to_int, f)

    data['Category'] = data['Category'].map(category_to_int)

    unique_labels = list(category_to_int.keys())
    
    # Extract texts
    texts = data['Text'].tolist()
    
    return data,texts, unique_labels

def preprocess_texts(texts, vocab_size=10000, max_seq_len=100,vocsize=100):
    """
    Tokenize and preprocess text data, returning padded sequences, tokenizer, and TF-IDF features.

    Parameters:
        texts (list): List of text data.
        vocab_size (int): Maximum vocabulary size for tokenizer.
        max_seq_len (int): Maximum sequence length for padding.

    Returns:
        tuple: (padded_sequences, tokenizer, tfidf_features, word_index, index_word)
    """
    # Tokenizing the sentences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_len)
    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)
    # Create TF-IDF features
    
    vectorizer = TfidfVectorizer(max_features=vocsize)
    tfidf_features = vectorizer.fit_transform(texts).toarray()
    
    # Build word index mappings
    word_index = tokenizer.word_index
    index_word = {i: word for word, i in word_index.items()}
    
    return padded_sequences, tokenizer, tfidf_features, word_index, index_word

def load_glove_embeddings(vocab_size, embedding_dim, word_index, glove_url=None):
    """
    Load GloVe embeddings and create an embedding matrix.

    Parameters:
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimension size.
        word_index (dict): Word index mapping.
        glove_url (str, optional): URL to download GloVe embeddings. Defaults to a pre-defined URL.

    Returns:
        np.array: Embedding matrix.
    """
    # Define GloVe URL if not provided
    if not glove_url:
        glove_url = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

    # Download and extract GloVe embeddings
    data_path = keras.utils.get_file("glove.6B.zip", glove_url, extract=True)
    data_path = os.path.dirname(data_path) + f'/glove.6B.{embedding_dim}d.txt'
    
    # Initialize embeddings index
    embeddings_index = {}
    with open(data_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    
    # Initialize embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i < vocab_size:  # Ensure index is within the vocab size
            if word in embeddings_index:
                embedding_matrix[i] = embeddings_index[word]
            else:
                embedding_matrix[i] = np.random.normal(size=(embedding_dim,))  # Random for OOV words

    return embedding_matrix

def pad_tfidf(X_tfidf, vocsize=100):
    """
    Pads or truncates the given TF-IDF features matrix to the specified vocabulary size (vocsize).
    
    Parameters:
    X_tfidf (numpy.ndarray): The input TF-IDF feature matrix.
    vocsize (int): The target vocabulary size (length).
    
    Returns:
    numpy.ndarray: The padded or truncated TF-IDF feature matrix.
    """
    if X_tfidf.shape[1] < vocsize:
        # Pad with zeros to the right
        padded_tfidf = np.pad(X_tfidf, 
                              ((0, 0), (0, vocsize - X_tfidf.shape[1])),
                              mode='constant')
    elif X_tfidf.shape[1] > vocsize:
        # Truncate to vocsize
        padded_tfidf = X_tfidf[:, :vocsize]
    
    return padded_tfidf