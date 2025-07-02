import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
print("Loading California housing dataset...")
housing = fetch_california_housing()

# Use only 1 feature - 'MedInc' (median income) as it's highly correlated with house prices
feature_index = 0  
x = housing.data[:, feature_index]  # type: ignore
y = housing.target  # type: ignore

x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

print(f"the shape of the inputs x is: {x.shape}")
print(f"the shape of the targets y is: {y.shape}")
print(f"Feature used: MedInc (Median Income)")

x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

del x_, y_

x_train = np.array(x_train)
x_cv = np.array(x_cv)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_cv = np.array(y_cv)
y_test = np.array(y_test)

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x.flatten(), y.flatten(), alpha=0.3, color='gray', s=10, label='All Data')
plt.title('Complete Dataset', fontweight='bold')
plt.xlabel('Median Income')
plt.ylabel('House Value ($100k)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(x_train.flatten(), y_train.flatten(), alpha=0.6, color='blue', s=15, label=f'Training Set ({len(x_train)} samples)')
plt.scatter(x_cv.flatten(), y_cv.flatten(), alpha=0.7, color='green', s=15, label=f'Cross-validation Set ({len(x_cv)} samples)')
plt.scatter(x_test.flatten(), y_test.flatten(), alpha=0.8, color='red', s=15, label=f'Test Set ({len(x_test)} samples)')
plt.title('Train/CV/Test Split', fontweight='bold')
plt.xlabel('Median Income')
plt.ylabel('House Value ($100k)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Data loaded and split successfully!")
print(f"📊 Feature: MedInc (Median Income)")
print(f"🎯 Ready for linear regression experiments!")
