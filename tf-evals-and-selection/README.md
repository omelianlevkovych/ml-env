# Machine Learning Model Comparison: Polynomial Regression vs Neural Networks

This project demonstrates **multiple machine learning approaches** for regression using the California Housing dataset. The implementation includes polynomial regression with cross-validation model selection and neural network architectures comparison, showcasing different approaches to the same regression problem.

## 🎯 Project Overview

This machine learning project showcases:
- **Polynomial Regression**: Feature engineering and systematic model selection via cross-validation
- **Neural Network Models**: Three different architectures with varying complexity
- **Model Comparison**: Direct performance comparison between traditional ML and deep learning
- **Proper ML Pipeline**: Train/validation/test splits with standardization
- **Performance Evaluation**: MSE comparison across different model types and complexities
- **Bias-Variance Trade-off**: Visualizing overfitting vs underfitting in both approaches

## 📁 Project Structure

```
tf-evals-and-selection/
├── main.py                    # Polynomial regression pipeline with model selection
├── neural-network.py          # Neural network models with TensorFlow/Keras
├── utils.py                   # MSE plotting utilities
├── requirements.txt           # Python dependencies
├── data/                      # Data directory
│   ├── california_housing.csv # California housing dataset
│   └── data_w3_ex1.csv       # Additional dataset
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy>=1.21.0` - Numerical operations
- `matplotlib>=3.5.0` - Plotting and visualization  
- `scikit-learn>=1.0.0` - ML algorithms and datasets
- `pandas>=1.3.0` - Data manipulation
- `tensorflow>=2.8.0` - Deep learning framework
- `keras>=2.8.0` - High-level neural network API

### 2. Run the Implementations

**Polynomial Regression Pipeline:**
```bash
python main.py
```

**Neural Network Models:**
```bash
python neural-network.py
```

## 🔬 Implementation Details

### Data Preparation (Both Approaches)
- **Dataset**: California Housing (20,640 samples)
- **Feature**: Median Income (MedInc) - single feature for clear visualization
- **Target**: House values in $100k units
- **Split**: 60% train / 20% validation / 20% test

## 📊 Approach 1: Polynomial Regression (`main.py`)

### Machine Learning Pipeline

1. **Data Loading & Splitting**
   ```python
   # Load California housing dataset
   housing = fetch_california_housing()
   x = housing.data[:, 0]  # MedInc feature only
   y = housing.target
   
   # 60/20/20 split
   x_train, x_cv, x_test, y_train, y_cv, y_test
   ```

2. **Feature Scaling**
   ```python
   # Standardize features (mean=0, std=1)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(x_train)
   ```

3. **Polynomial Feature Engineering**
   ```python
   # Create polynomial features up to degree 10
   poly = PolynomialFeatures(degree=degree, include_bias=False)
   X_train_mapped = poly.fit_transform(x_train)
   ```

4. **Model Training & Evaluation**
   ```python
   # Train linear regression on polynomial features
   model = LinearRegression()
   model.fit(X_train_mapped_scaled, y_train)
   
   # Evaluate on train and validation sets
   train_mse = mean_squared_error(y_train, yhat) / 2
   cv_mse = mean_squared_error(y_cv, yhat) / 2
   ```

5. **Model Selection**
   ```python
   # Choose degree with lowest validation MSE
   best_degree = np.argmin(cv_mses) + 1
   
   # Final evaluation on test set
   test_mse = mean_squared_error(y_test, yhat) / 2
   ```

### Expected Output (Polynomial Regression)

```
Loading California housing dataset...
the shape of the inputs x is: (20640, 1)
the shape of the targets y is: (20640, 1)
Feature used: MedInc (Median Income)

the shape of the training set (input) is: (12384, 1)
the shape of the training set (target) is: (12384, 1)

the shape of the cross validation set (input) is: (4128, 1)
the shape of the cross validation set (target) is: (4128, 1)

the shape of the test set (input) is: (4128, 1)
the shape of the test set (target) is: (4128, 1)

✅ Data loaded and split successfully!
📊 Feature: MedInc (Median Income)
🎯 Ready for linear regression experiments!

training MSE (using sklearn function): [value]
training MSE (for-loop implementation): [value]
Cross validation MSE: [value]

Lowest CV MSE is found in the model with degree=[X]
Training MSE: [value]
Cross Validation MSE: [value]  
Test MSE: [value]
```

## 🧠 Approach 2: Neural Networks (`neural-network.py`)

### Neural Network Architectures

The implementation compares three different neural network architectures:

1. **Model 1 (Simple)**: 
   - Input → Dense(25, ReLU) → Dense(15, ReLU) → Dense(1, Linear)
   - 2 hidden layers, moderate complexity

2. **Model 2 (Deep)**: 
   - Input → Dense(20, ReLU) → Dense(12, ReLU) → Dense(12, ReLU) → Dense(20, ReLU) → Dense(1, Linear)
   - 4 hidden layers, deeper architecture

3. **Model 3 (Complex)**: 
   - Input → Dense(32, ReLU) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(4, ReLU) → Dense(12, ReLU) → Dense(1, Linear)
   - 5 hidden layers, varying widths

### Neural Network Pipeline

1. **Data Preprocessing**
   ```python
   # Apply polynomial features (degree=1, so just scaling)
   poly = PolynomialFeatures(degree=1, include_bias=False)
   X_train_mapped = poly.fit_transform(x_train)
   
   # Standardize features
   scaler = StandardScaler()
   X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
   ```

2. **Model Creation**
   ```python
   def build_models():
       # Define three different architectures
       model1 = keras.Sequential([...])  # Simple
       model2 = keras.Sequential([...])  # Deep  
       model3 = keras.Sequential([...])  # Complex
       return models
   ```

3. **Training & Evaluation**
   ```python
   # Compile with MSE loss and Adam optimizer
   model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.1))
   
   # Train for 300 epochs
   model.fit(X_train_mapped_scaled, y_train, epochs=300, verbose=0)
   
   # Evaluate on train and validation sets
   train_mse = mean_squared_error(y_train, yhat) / 2
   cv_mse = mean_squared_error(y_cv, yhat) / 2
   ```

### Expected Output (Neural Networks)

```
Loading California housing dataset...
✅ Data loaded and split successfully!
📊 Feature: MedInc (Median Income)
🎯 Ready for linear regression experiments!

Training Model_1...
Done!

Training Model_2...
Done!

Training Model_3...
Done!

RESULTS:
Model 1: Training MSE: [value], CV MSE: [value]
Model 2: Training MSE: [value], CV MSE: [value]
Model 3: Training MSE: [value], CV MSE: [value]
```

## 🛠 Utility Functions (`utils.py`)

### MSE Plotting
```python
import utils

# Plot training and validation MSEs vs polynomial degrees
utils.plot_train_cv_mses(degrees, train_mses, cv_mses, 
                        title="Degree of polynomial vs. train and CV MSEs")
```

**Function**: `plot_train_cv_mses()`
- **Purpose**: Visualize bias-variance trade-off
- **Inputs**: Polynomial degrees, training MSEs, validation MSEs, optional title
- **Output**: Line plot showing both MSE curves for model selection
- **Features**: Formatted with markers, colors, grid, and legend

## 📊 Key Machine Learning Concepts Demonstrated

### 1. **Polynomial Regression vs Neural Networks**
- **Polynomial**: Explicit feature engineering with linear models
- **Neural Networks**: Automatic feature learning through hidden layers
- **Comparison**: Traditional ML vs Deep Learning approaches

### 2. **Model Selection Strategies**
- **Polynomial**: Cross-validation across different polynomial degrees
- **Neural Networks**: Architecture comparison (depth vs width)
- **Both**: Systematic evaluation using train/validation/test methodology

### 3. **Feature Scaling**
- Standardizes features before model training
- Critical for both polynomial regression and neural networks
- Ensures numerical stability and fair comparison

### 4. **Architecture Exploration (Neural Networks)**
- **Model 1**: Baseline architecture with moderate complexity
- **Model 2**: Deeper network testing depth hypothesis
- **Model 3**: Complex architecture with varying layer widths

### 5. **Proper ML Evaluation**
- Train set: Model fitting
- Validation set: Hyperparameter tuning and model selection
- Test set: Final unbiased performance estimate

## 🎯 Learning Outcomes

After running both implementations, you'll understand:

✅ **Polynomial Regression Concepts:**
- Polynomial feature engineering and its effects
- Cross-validation for model selection
- Bias-variance trade-off visualization
- Linear regression with engineered features

✅ **Neural Network Concepts:**
- Different neural network architectures
- Deep learning with TensorFlow/Keras
- Architecture complexity vs performance
- Automatic feature learning

✅ **Model Comparison:**
- Traditional ML vs Deep Learning approaches
- When to use polynomial regression vs neural networks
- Performance trade-offs between approaches
- Complexity vs interpretability

✅ **Practical Skills:**
- Complete scikit-learn ML workflow
- TensorFlow/Keras neural network implementation
- Systematic model evaluation and comparison
- Professional ML pipeline development

## 🚀 Extensions and Experiments

### Try Different Features
```python
# Experiment with other housing features
feature_index = 1  # HouseAge
feature_index = 2  # AveRooms
# Or combine multiple features for both approaches
```

### Regularization
```python
# Polynomial Regression
from sklearn.linear_model import Ridge, Lasso

# Neural Networks  
model.add(keras.layers.Dropout(0.2))  # Dropout
# Or L1/L2 regularization in layer definitions
```

### Architecture Variations
```python
# Try different neural network configurations
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
```

### Advanced Comparisons
```python
# Compare more polynomial degrees
for degree in range(1, 15):

# Try different optimizers for neural networks
keras.optimizers.SGD(learning_rate=0.01)
keras.optimizers.RMSprop(learning_rate=0.001)
```

This project provides comprehensive coverage of both traditional machine learning (polynomial regression) and modern deep learning approaches (neural networks) for regression problems, demonstrating the evolution and trade-offs in machine learning methodologies. 