import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from autils import plot_digit, plot_sample_digits
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

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

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1)

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((8,8)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Display the label above the image
        ax.set_title(y[random_index])
        ax.set_axis_off()
    
    plt.show()

    model = Sequential(
        [               
            keras.Input(shape=(64,)),    #specify input size - 8x8 = 64 features
            Dense(25, activation='sigmoid'),
            Dense(15, activation='sigmoid'),
            Dense(1, activation='sigmoid')
        ], name = "my_model" 
    )                            

    model.summary()

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer='adam',
    )

    model.fit(
        X, y,
        epochs=20
    )

    # Predict for the first zero
    zero_indices = np.where(y == 0)[0]
    if len(zero_indices) > 0:
        idx_zero = zero_indices[0]
        prediction_zero = model.predict(X[idx_zero].reshape(1, 64))
        print(f" predicting a zero: {prediction_zero}")
        yhat_zero = 1 if prediction_zero >= 0.5 else 0
        print(f"prediction after threshold (zero): {yhat_zero}")
    else:
        print("No samples with label 0 found.")

    # Predict for the first one
    one_indices = np.where(y == 1)[0]
    if len(one_indices) > 0:
        idx_one = one_indices[0]
        prediction_one = model.predict(X[idx_one].reshape(1, 64))
        print(f" predicting a one:  {prediction_one}")
        yhat_one = 1 if prediction_one >= 0.5 else 0
        print(f"prediction after threshold (one): {yhat_one}")
    else:
        print("No samples with label 1 found.")

if __name__ == "__main__":
    main()

