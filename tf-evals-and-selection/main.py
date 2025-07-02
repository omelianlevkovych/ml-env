import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
X_train_scaled = scaler_linear.fit_transform(x_train)


# Train the model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Eval the model
yhat = linear_model.predict(X_train_scaled)
# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

total_squared_error = 0

for i in range(len(yhat)):
    squared_error_i  = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i                                              

mse = total_squared_error / (2*len(yhat))
print(f"training MSE (for-loop implementation): {mse}")

# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_scaled = scaler_linear.transform(x_cv)

# Feed the scaled cross validation set
yhat = linear_model.predict(X_cv_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")


# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_mapped = poly.fit_transform(X_train_scaled)
print(X_train_mapped[:5])

sclaer_poly = StandardScaler()
X_train_mapped_scaled = sclaer_poly.fit_transform(X_train_mapped)
print(X_train_mapped_scaled[:5])

model = LinearRegression()
model.fit(X_train_mapped_scaled, y_train)
yhat = model.predict(X_train_mapped_scaled)
print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2}")

X_cv_mapped = poly.transform(x_cv)
X_cv_mapped_scaled = sclaer_poly.transform(X_cv_mapped)

yhat = model.predict(X_cv_mapped_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")

# We can have a loop and try different degrees
train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.
for degree in range(1,11):
    
    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)
    
    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train )
    models.append(model)
    
    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)
    
    # Add polynomial features and scale the cross validation set
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
    
    # Compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)
    
# Plot the results
degrees = range(1, 11)
utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="Degree of polynomial vs. train and CV MSEs")

# Choosing the best model
degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")

X_test_mapped = polys[degree-1].transform(x_test)

# Scale the test set
X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

# Compute the test MSE
yhat = models[degree-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")