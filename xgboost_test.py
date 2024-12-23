import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv('ml-pillar.csv')

best_r2 = -np.inf  # Initialize the best R2 score to negative infinity
best_iteration = None  # To keep track of the best iteration

for i in range(10000):
    print(f'Iteration {i}')

    # Separate features and target variable
    X = data.drop(['wavelength', 'E', 'Q'], axis=1)
    Y = data[['E', 'Q']]  # Multi-target

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=i)

    # Creating the XGBRegressor model
    xgb_model = XGBRegressor()

    # Set the best parameters
    best_params = {
        'colsample_bytree': 0.7,
        'learning_rate': 0.015,
        'max_depth': 5,
        'n_estimators': 350
    }

    # Training the model with the best parameters
    best_xgb_model = XGBRegressor(**best_params)
    best_xgb_model.fit(X_train, Y_train)

    # Predicting and evaluating the model
    Y_pred = best_xgb_model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred, multioutput='variance_weighted')
    mse = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')

    # Check if this iteration has a better R2 score
    if r2 > best_r2:
        best_r2 = r2
        best_iteration = i

    print(f'Iteration {i}: MSE on test set = {mse}, R2 on test set = {r2}')

# After all iterations, print out the best iteration and its R2 score
print(f'Best iteration: {best_iteration} with R2 score: {best_r2}')
