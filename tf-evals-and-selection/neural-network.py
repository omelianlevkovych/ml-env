import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import utils

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

degree = 1
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)

scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)

# Build and train the model
nn_train_mses = []
nn_cv_mses = []

# Build the models
def build_models():
    models = []
    
    # Model 1: 25 units -> 15 units -> 1 unit
    model1 = keras.Sequential([
        keras.layers.Dense(25, activation='relu', input_shape=(X_train_mapped_scaled.shape[1],)),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ], name='Model_1')
    models.append(model1)
    
    # Model 2: 20 units -> 12 units -> 12 units -> 20 units -> 1 unit
    model2 = keras.Sequential([
        keras.layers.Dense(20, activation='relu', input_shape=(X_train_mapped_scaled.shape[1],)),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ], name='Model_2')
    models.append(model2)
    
    # Model 3: 32 units -> 16 units -> 8 units -> 4 units -> 12 units -> 1 unit
    model3 = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X_train_mapped_scaled.shape[1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ], name='Model_3')
    models.append(model3)
    
    return models

nn_models = build_models()

# Loop over the models
for model in nn_models:
    # Setup the loss and optimizer
    model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    )

    print(f"Training {model.name}...")
    
    # Train the model
    model.fit(
        X_train_mapped_scaled, y_train,
        epochs=300,
        verbose=0
    )
    
    print("Done!\n")

    
    # Record the training MSEs
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    nn_train_mses.append(train_mse)
    
    # Record the cross validation MSEs 
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    nn_cv_mses.append(cv_mse)

    
# print results
print("RESULTS:")
for model_num in range(len(nn_train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
        f"CV MSE: {nn_cv_mses[model_num]:.2f}"
        )