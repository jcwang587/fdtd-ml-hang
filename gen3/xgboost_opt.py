import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

random.seed(42)

# Load the data
data = pd.read_csv("data/fe_merged.csv")

# Separate features and target variable
X = data.drop(columns=["FE"])
y = data["FE"]

# Normalize the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

# Creating the XGBRegressor model
xgb_model = XGBRegressor()

param_grid = {
    "n_estimators": [1000],
    "max_depth": [4],
    "learning_rate": [0.01],
}

# Set up the grid search
grid_search = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=3,  # Using 3-fold cross-validation
    scoring="r2",
    verbose=1,  # Shows progress
    n_jobs=-1,  # Use all available cores
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Re-train the model with the best parameters
best_params = grid_search.best_params_
optimized_xgb_model = XGBRegressor(**best_params)
optimized_xgb_model.fit(X_train, y_train)

# Predicting and evaluating the model with optimized parameters
y_pred_optimized = optimized_xgb_model.predict(X_test)
r2_optimized = r2_score(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)

print(f"R2 on test set for E: {r2_optimized:.4f}, MSE: {mse_optimized:.4f}")

# Round correlation values to 2 decimal places and set very small values to 0
corr_matrix = X_scaled.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.01] = 0
np.fill_diagonal(corr_matrix.values, 1.0)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Plot the correlation matrix with the mask
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", mask=mask, fmt=".2f")

# set tight layout
plt.tight_layout()

plt.savefig("./corr_matrix.png", dpi=600, format="png")
plt.close()
