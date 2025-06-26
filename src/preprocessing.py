import pickle
import numpy as np
from scipy.stats import mode
from scipy.signal import find_peaks

# Load subject data from WESAD .pkl file
def load_subject_data(file_path):
    """
    Loads a WESAD subject's data from a .pkl file.
    Handles compatibility with Python 2 pickle encoding.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # required due to Python 2 format
    return data

# Feature Extraction from signal windows
def extract_features(signal, labels, window_size=100):
    """
    Extracts statistical and shape-based features from fixed-size signal windows.
    Also assigns a label to each window using the most frequent label in the window.
    
    Returns:
        - features: Array of extracted feature vectors
        - targets: Array of corresponding window labels
    """
    # Convert inputs to NumPy arrays
    signal = np.array(signal)
    labels = np.array(labels)

    features, targets = [], []

    # Iterate over signal with fixed-size sliding window
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i + window_size]
        label_window = labels[i:i + window_size]

        # Extract basic statistical features from the signal window
        mean = np.mean(window)
        std = np.std(window)
        minimum = np.min(window)
        maximum = np.max(window)

        # Calculate slope as difference from start to end of the window
        slope = (window[-1] - window[0]) / window_size

        # Count the number of peaks in the window
        peaks, _ = find_peaks(window)
        peak_count = len(peaks)

        # Append feature vector
        features.append([mean, std, minimum, maximum, slope, peak_count])

        # Determine the most frequent label in the current window
        dominant_label = mode(label_window, keepdims=True).mode[0]
        targets.append(dominant_label)

    return np.array(features), np.array(targets)

# Downsample labels using dominant (mode) values 
def downsample_labels_by_mode(labels, target_length):
    """
    Downsamples a label array to match a target length by aggregating
    each chunk with the most frequent (mode) label.

    Useful for aligning label length with reduced signal length.
    """
    factor = len(labels) // target_length

    # Aggregate labels in chunks and take the mode of each chunk
    downsampled = [
        mode(labels[i:i + factor], keepdims=True).mode[0]
        for i in range(0, len(labels), factor)
        if len(labels[i:i + factor]) > 0
    ]

    # Ensure the output has exactly the desired length
    return np.array(downsampled[:target_length])
