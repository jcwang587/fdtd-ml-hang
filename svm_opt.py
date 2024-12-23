from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler


# Load the data
data = pd.read_csv('data.csv')
data_shuffled = data.sample(frac=1, random_state=411)

# split the data to train, validation and test
train_idx = data_shuffled.index[:int(0.9 * len(data_shuffled))]
valid_idx = data_shuffled.index[int(0.9 * len(data_shuffled)):]
test_idx = data_shuffled.index[int(0.9 * len(data_shuffled)):]

# split the data to train, validation and test, the first 5 columns are features, the last column is the target
X_train = data_shuffled.iloc[train_idx, :-2]
y_train = data_shuffled.iloc[train_idx, -1]

X_valid = data_shuffled.iloc[valid_idx, :-2]
y_valid = data_shuffled.iloc[valid_idx, -1]

X_test = data_shuffled.iloc[test_idx, :-2]
y_test = data_shuffled.iloc[test_idx, -1]

# Create the model
svr = SVR()

# Define the hyperparameters
C = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
gamma = [float(x) for x in np.linspace(start=0.1, stop=1, num=10)]
kernel = ['linear', 'poly', 'rbf']
degree = [1, 2, 3, 4, 5]

# Create the random grid
random_grid = {'C': C,
               'gamma': gamma,
               'kernel': kernel,
               'degree': degree}

# Hyperparameter tuning using validation set
min_mse = float('inf')
best_params = None
iteration = 0

# It's recommended to use a limited number of iterations for practical runtime consideration
for params in ParameterSampler(random_grid, n_iter=100, random_state=587):
    iteration += 1
    svr.set_params(**params)
    svr.fit(X_train, y_train)
    preds = svr.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)

    print(f"Iteration {iteration}: Params={params}, MSE={mse}")

    if mse < min_mse:
        min_mse = mse
        best_params = params

print('Best parameters found: ', best_params)

# predict on test set
svr.set_params(**best_params)
svr.fit(X_train, y_train)
predicted = svr.predict(X_test)
mse = mean_squared_error(y_test, predicted)
r2 = svr.score(X_test, y_test)
print('MSE on test set: ', mse)
print('R2 on test set: ', r2)

# save the results
rf_res = pd.DataFrame({'Idx': list(test_idx), 'E': list(y_test), 'E_pred': list(predicted)})
rf_res.to_csv('./csv_data/svr_opt_res.csv', index=False)

# save the model
import pickle
pickle.dump(svr, open('./model_output/svr_opt_model.pkl', 'wb'))
