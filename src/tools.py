from pathlib import Path
import numpy as np
import tensorflow as tf
import math
import psutil
from collections import Counter
import os
import time

""" Contains a few generic helper functions
"""

MAX_ENTROPY_NORMALIZED = 255 / 128.0
HALF_MAX_LENGTH = 128 / 2.0

def read_file(file_path: str) -> bytes:
    """
    Reads a file and returns its content as a byte string (bytes object).

    :param file_path: Path to the file to be read.
    :return: Byte string (bytes object) containing the file content.
    """
    with open(file_path, 'rb') as file:
        byte_string = file.read()
    return byte_string

def count_unique(byte_string: bytes) -> int:
    """
    Counts the number of unique characters (bytes) in a byte string.

    :param byte_string: Input byte string (bytes object).
    :return: Number of unique characters (bytes) in the byte string.
    """
    return len(set(byte_string))

def get_characters(byte_string: bytes) -> bytes:
    """
    Returns a byte string containing each unique character (byte) from the input exactly once.

    :param byte_string: Input byte string (bytes object).
    :return: Byte string with each unique character (byte) from the input.
    """
    unique_bytes = sorted(set(byte_string))
    return bytes(unique_bytes)

def read_files_to_byte_strings(directory: str):
    """
    Reads all files in a directory into byte strings.

    :param directory: Path to the directory containing the files.
    :return: A list of list of byte strings.
    """
    file_contents = []
    dir_path = Path(directory)
    # Iterate over all files in the directory
    for file_path in dir_path.iterdir():
        if file_path.is_file(): 
            with open(file_path, 'rb') as file:
                x = file.read().splitlines()
                file_contents.append(x)
    return file_contents

def capitalize_byte_string(byte_string: bytes) -> bytes:
    """
    Capitalizes a byte string.

    :param byte_string: Input byte string (bytes object).
    :return: Capitalized byte string.
    """
    # Decode the byte string to a regular string
    string = byte_string.decode('utf-8')
    # Capitalize the string
    capitalized_string = string.capitalize()
    # Encode the string back to a byte string
    return capitalized_string.encode('utf-8')

def calculate_min_bits(input_string: str) -> int:
    """ 
    Calculates the minimum amount of bits required to represent the string
    """
    # Calculate the frequency of each character in the string
    freq = Counter(input_string)
    
    # Length of the string
    string_length = len(input_string)
    
    # Convert frequency counts to a NumPy array
    counts = np.array(list(freq.values()))
    
    # Calculate probabilities
    probabilities = counts / string_length
    
    # Calculate entropy per symbol using vectorized operations
    log_probabilities = np.log2(probabilities)
    entropy_per_symbol = -np.sum(probabilities * log_probabilities)
    
    # Total entropy in bits for the entire string
    total_entropy_bits = entropy_per_symbol * string_length
    
    # Round up to the nearest whole number (since bits are discrete)
    min_bits_required = np.ceil(total_entropy_bits)
    
    return int(min_bits_required)

def get_entropy(s: str) -> float:
    """
    Calculate the entropy value of a string.
    """
    x = calculate_min_bits(s)
    return min(x / 127.0, MAX_ENTROPY_NORMALIZED) - 1.0

def normalize_length(length):
    """
    Normalize length to the range [-1, 1].
    max_length is the maximum possible length of the input.
    """
    return length / HALF_MAX_LENGTH - 1.0

def preprocess_password(password, max_length=128):
    """
    Normalize a password string to its ASCII values and pad/truncate it to max_length.
    """
    # Precompute constants
    NORMALIZATION_FACTOR = 128.0
    PADDING_VALUE = -1.0

    # Calculate minimum amount of bits required to represent this password
    entropy = get_entropy(password)

    # Convert characters to ASCII values using NumPy for faster conversion
    ascii_values = np.frombuffer(password.encode('utf-8'), dtype=np.uint8)

    # Normalize values to [-1, 1]
    normalized = ascii_values / NORMALIZATION_FACTOR - 1.0

    # Calculate and normalize length
    normalized_length = normalize_length(len(password))

    # Add entropy and length as first features
    normalized = np.insert(normalized, 0, [entropy, normalized_length])

    # Pad or truncate to max_length
    if len(normalized) < max_length:
        # Pad with PADDING_VALUE
        padded = np.pad(normalized, (0, max_length - len(normalized)), mode='constant', constant_values=PADDING_VALUE)
    else:
        # Truncate to max_length
        padded = np.clip(normalized, None, max_length)[:max_length]

    return padded

def is_not_fully_masked(sequence):
    """
    Returns `True` if the sequence (excluding the first two items) is not fully masked.
    """
    return tf.reduce_any(sequence[2:] != -1.0)

# Create a tf.data.Dataset pipeline
def create_dataset(file_path, max_length=128, compression_type=""):
    """
    Create a tf.data.Dataset pipeline that reads passwords from a file, preprocesses them,
    and returns batches of data.
    """
    # Read the file line by line
    dataset = tf.data.TextLineDataset(file_path, compression_type=compression_type)
    
    # Preprocess each password
    def preprocess(line):
        password = line.numpy().decode('utf-8').strip()
        preprocessed = preprocess_password(password, max_length)
        return preprocessed

    # Use tf.py_function to apply the preprocessing function
    dataset = dataset.map(lambda x: tf.py_function(preprocess, [x], tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Reshape the data to match the model's input shape
    dataset = dataset.map(lambda x: tf.reshape(x, (max_length, 1)), num_parallel_calls=tf.data.AUTOTUNE)
 
    # Filter out fully masked inputs
    # If any of the datasets has an empty line, it will cause cudnn to crash
    dataset = dataset.filter(is_not_fully_masked)

    return dataset

def add_labels(dataset, label):
    """
    Add labels to the dataset and reshape them to match the model's output shape.
    """
    # Create a dataset of labels with the desired shape (1,)
    labels_dataset = tf.data.Dataset.from_tensors(tf.reshape(label, (1,))).repeat()

    # Zip the input dataset with the labels dataset
    labeled_dataset = tf.data.Dataset.zip((dataset, labels_dataset))

    # Prefetch for better performance
    labeled_dataset = labeled_dataset.prefetch(tf.data.AUTOTUNE)
    return labeled_dataset

def assign_split(x, y, train_ratio=0.8):
    """
    Assigns each sample to train or validation using a mask-based approach.
    """
    # Generate a random number for the entire batch
    rng = tf.random.uniform(shape=(), dtype=tf.float32)

    # Use a mask to assign the split label
    split_label = tf.cond(
        rng < train_ratio,
        lambda: tf.constant("train", dtype=tf.string),
        lambda: tf.constant("val", dtype=tf.string)
    )

    return x, y, split_label

def positional_encoding(length, depth):
    """ 
    Generates position encoding for transformer.
    """
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis] 
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float16)

def log_memory_usage():
    """ 
    Returns memory usage.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Memory usage in MB

def measure_iteration(ds):
    """
    Interates all the data from a dataset, and prints statistics
    """
    start_time = time.time()
    x = 0
    for batch in ds:
        x += 1
        print (f"Item number: {x}", end='\r')

    print(f"Final memory usage: {log_memory_usage():.2f} MB")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    exit(0)

def serialize_data(features, values):
    """ 
    Serializes data into tfrecord compatible string.
    """
    # Ensure the data is in the correct shape and type
    features = tf.ensure_shape(features, (64, 1))  # Enforce shape (64, 1)
    features = tf.cast(features, tf.float32)  # Ensure float32 type
    values = tf.ensure_shape(values, (1,))  # Enforce shape (1,)
    values = tf.cast(values, tf.float32)  # Ensure float32 type

    # Serialize the tensors
    serialized_features = tf.io.serialize_tensor(features)
    serialized_values = tf.io.serialize_tensor(values)

    # Create a feature dictionary
    feature = {
        "features": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_features.numpy()])),
        "values": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_values.numpy()])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def save_to_tfrecord(dataset, filename):
    """
    Saves dataset into tfrecord file.
    """
    x = 0
    with tf.io.TFRecordWriter(filename) as writer:
        for features, values in dataset:
            example = serialize_data(features, values)
            writer.write(example)
            x += 1
            print(f"Written records: {x}", end="\r")
    print ("\nFile finished")

def parse_example(example_proto):
    """
    Parses tfrecords.
    """
    feature_description = {
        "features": tf.io.FixedLenFeature([], tf.string),  # Serialized features
        "values": tf.io.FixedLenFeature([], tf.string),    # Serialized values
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize features and values
    features = tf.io.parse_tensor(example["features"], out_type=tf.float32)
    values = tf.io.parse_tensor(example["values"], out_type=tf.float32)
    
    # Ensure the shapes are set correctly
    features = tf.ensure_shape(features, (64, 1))  # Enforce shape (64, 1)
    values = tf.ensure_shape(values, (1,))         # Enforce shape (1,)
    
    return features, values

def load_tfrecord(filename):
    """
    Loads dataset from tfrecord.
    """
    raw_dataset = tf.data.TFRecordDataset(filename)
    return raw_dataset.map(parse_example)

