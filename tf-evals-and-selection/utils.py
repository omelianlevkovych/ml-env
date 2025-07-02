import matplotlib.pyplot as plt
import numpy as np

def plot_train_cv_mses(degrees, train_mses, cv_mses, title="Training and CV MSEs"):
    """
    Plot training and cross-validation MSEs vs polynomial degrees.
    
    Args:
        degrees: Range of polynomial degrees
        train_mses: Training MSE values
        cv_mses: Cross-validation MSE values  
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_mses, marker='o', color='blue', label='Training MSE', linewidth=2)
    plt.plot(degrees, cv_mses, marker='s', color='red', label='Cross-validation MSE', linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() 