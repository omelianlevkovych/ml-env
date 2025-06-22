import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary for a binary classification model.
    
    Args:
        model: trained model with predict method
        X: input features
        y: true labels
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Make predictions on the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

def plot_loss_history(history):
    """
    Plot training and validation loss history.
    
    Args:
        history: Keras training history object
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_digit(image, title="Digit"):
    """
    Plot a single digit image.
    
    Args:
        image: flattened image array (should be 64 elements for 8x8 digits)
        title: title for the plot
    """
    # Reshape to 8x8 for digits dataset
    if len(image) == 64:
        image = image.reshape(8, 8)
    elif len(image) == 400:
        image = image.reshape(20, 20)
    
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_sample_digits(X, y, n_samples=5):
    """
    Plot sample digits from the dataset.
    
    Args:
        X: input features
        y: labels
        n_samples: number of samples to plot
    """
    fig, axes = plt.subplots(1, n_samples, figsize=(2*n_samples, 2))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(X))
        image = X[idx]
        
        # Reshape to 8x8 for digits dataset
        if len(image) == 64:
            image = image.reshape(8, 8)
        elif len(image) == 400:
            image = image.reshape(20, 20)
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {y[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and print metrics.
    
    Args:
        model: trained model
        X_test: test features
        y_test: test labels
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred_classes)
    print(f"Confusion Matrix:\n{cm}")
    
    # Classification report
    print(f"Classification Report:\n{classification_report(y_test, y_pred_classes)}")
    
    return accuracy, y_pred_classes 