# Linear Regression with California Housing Dataset

This project provides a foundation for practicing linear regression using the California Housing dataset. The codebase includes data loading, preprocessing, visualization utilities, and train/validation/test splitting.

## 📁 Project Structure

```
tf-evals-and-selection/
├── main.py                    # Main script - data loading and visualization
├── utils.py                   # Plotting and utility functions for regression analysis
├── requirements.txt           # Python dependencies
├── data/                      # Data directory
│   ├── data_w3_ex1.csv       # Synthetic dataset (100 samples, 2 features)
│   └── california_housing.csv # California housing dataset (20,640 samples)
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

The dependencies are:
- `numpy>=1.21.0` - Numerical operations
- `matplotlib>=3.5.0` - Plotting and visualization
- `scikit-learn>=1.0.0` - Machine learning datasets and utilities
- `pandas>=1.3.0` - Data manipulation (optional)

### 2. Run the Data Loading Script

```bash
python main.py
```

This will:
- Load the California housing dataset from scikit-learn
- Use the median income feature (MedInc) as the single input feature
- Split the data into training (60%), cross-validation (20%), and test (20%) sets
- Display the data shapes and create visualizations showing the complete dataset and the train/cv/test split

## 📊 Current Implementation

### Main Script (`main.py`)

The main script demonstrates:

1. **Data Loading**: Fetches the California housing dataset using scikit-learn
2. **Feature Selection**: Uses only the 'MedInc' (median income) feature as it's highly correlated with house prices
3. **Data Preprocessing**: Reshapes 1D arrays to 2D for compatibility with sklearn
4. **Train/CV/Test Split**: Splits data into 60%/20%/20% for training, cross-validation, and testing
5. **Visualization**: Creates side-by-side plots showing:
   - Complete dataset scatter plot
   - Train/CV/Test split with different colors for each set

### Expected Output

When you run `python main.py`, you'll see:

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
```

## 🛠 Utility Functions (`utils.py`)

The utility module provides comprehensive plotting functions for regression analysis:

### Basic Dataset Plotting
```python
import utils

# Plot any dataset
utils.plot_dataset(x, y, title="Your Dataset", xlabel="Input", ylabel="Target")
```

### Comprehensive Regression Results
```python
# Plot training data, regression line, test predictions, and residuals
utils.plot_regression_results(x_train, y_train, x_test, y_test, y_train_pred, y_test_pred, model)
```

### Prediction vs Actual Comparison
```python
# Compare predicted vs actual values
utils.plot_prediction_vs_actual(y_true, y_pred, title="Model Performance")
```

### Feature Importance (for multiple features)
```python
# Visualize feature importance in multiple regression
utils.plot_feature_importance(feature_names, coefficients, title="Feature Importance")
```

### Dataset Information
```python
# Print comprehensive dataset statistics
utils.print_dataset_info(x, y, feature_names=["MedInc"])
```

## 📈 Available Datasets

### 1. California Housing Dataset (`california_housing.csv`)
- **Size**: 20,640 samples
- **Features**: 8 features available (currently using 1: MedInc)
- **Target**: Median house value (in $100k)
- **Source**: Real-world data from sklearn
- **Current Usage**: Single feature regression with median income

### 2. Synthetic Dataset (`data_w3_ex1.csv`)
- **Size**: 100 samples
- **Features**: Simple 2-column format
- **Usage**: Available for simple experiments

### California Housing Features Available:
- `MedInc`: Median income in block group (currently used)
- `HouseAge`: Median house age in block group  
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

## 🔧 Next Steps - Extend the Implementation

The current codebase provides the foundation. Here are natural next steps:

### 1. Add Linear Regression Model
```python
from sklearn.linear_model import LinearRegression

# Train a model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Use utils for visualization
utils.plot_regression_results(x_train, y_train, x_test, y_test, y_train_pred, y_test_pred, model)
```

### 2. Add Model Evaluation
```python
from sklearn.metrics import mean_squared_error, r2_score

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
```

### 3. Experiment with Multiple Features
```python
# Use multiple features instead of just MedInc
x_multi = housing.data[:, [0, 1, 2]]  # MedInc, HouseAge, AveRooms
```

### 4. Add Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
```

## 🎯 Current Capabilities

✅ **Implemented:**
- Data loading and preprocessing
- Train/validation/test splitting
- Comprehensive visualization utilities
- Dataset information and statistics
- Professional plotting functions for regression analysis

🔄 **Ready to Add:**
- Linear regression model training
- Model evaluation metrics
- Cross-validation analysis
- Multiple feature regression
- Polynomial regression
- Regularization techniques

## 💡 Usage Tips

1. **Start with the current implementation** to understand data structure
2. **Use the utility functions** for consistent, professional visualizations
3. **The data is already preprocessed** and ready for model training
4. **Cross-validation set is prepared** for hyperparameter tuning
5. **All plotting functions are ready** for regression analysis

The foundation is solid - you can focus on implementing and experimenting with different regression techniques! 🚀 