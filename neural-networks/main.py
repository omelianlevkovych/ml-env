import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from autils import plot_digit, plot_sample_digits

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def load_data():
    """
    Load handwritten digits dataset (0 and 1 only)
    Returns:
        X: numpy array of shape (1000, 64) - flattened 8x8 pixel images
        y: numpy array of shape (1000,) - labels (0 or 1)
    """
    # Load the digits dataset
    X_all, y_all = load_digits(return_X_y=True)
    
    # Filter for only digits 0 and 1
    mask = (y_all == 0) | (y_all == 1)
    X_filtered = X_all[mask]
    y_filtered = y_all[mask]
    
    # Take first 1000 samples (or all if less than 1000)
    n_samples = min(1000, len(X_filtered))
    X = X_filtered[:n_samples]
    y = y_filtered[:n_samples]
    
    # Normalize pixel values to [0, 1] range
    X = X / 16.0  # digits dataset uses 0-16 scale
    
    return X, y

def main():
    """Main function to run the digit classification demo"""
    # Load the data
    X, y = load_data()
    print('The first element of X is: ', X[0])
    print(f'Dataset shape: X={X.shape}, y={y.shape}')
    
    print('\nDisplaying sample digits:')
    plot_sample_digits(X, y, n_samples=3)

if __name__ == "__main__":
    main()

