import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv("./csv_data/dataset.csv")

# Separate features and target variable
X = data.iloc[:, :-1]
y = data["fe"]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    # X_scaled, y, test_size=0.1, random_state=2546
    X_scaled, y, test_size=0.1, random_state=2737
)

# Creating the XGBRegressor model
xgb_model = XGBRegressor()

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.015, 0.02],
}

# Set up the grid search
grid_search = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=3,  # Using 3-fold cross-validation
    scoring='r2',
    verbose=1,  # Shows progress
    n_jobs=-1  # Use all available cores
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

# Plot the parity plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_optimized, s=150)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "k--")
plt.xlabel("Actual E", fontsize=32)
plt.ylabel("Predicted E", fontsize=32)
plt.title("Single-Target XGBoost Prediction for E\n$R^2$: {:.3f}, MSE: {:.3f}".format(r2_optimized, mse_optimized), fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("./parity_plot_E.svg", dpi=1200, format="svg")
plt.close()

# shapley values
import shap

explainer = shap.TreeExplainer(optimized_xgb_model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, show=False, feature_names=X.columns)

fig, ax = plt.gcf(), plt.gca()
fig.set_size_inches(6, 4)

ax.set_xlabel("SHAP value", fontsize=15)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.axes[-1].yaxis.label.set_size(15)
fig.axes[-1].get_yticklabels()[0].set_fontsize(15)
fig.axes[-1].get_yticklabels()[-1].set_fontsize(15)

for label in ax.get_yticklabels():
    label.set_fontsize(15)

# add text in the right bottom corner
plt.text(28, 0.1, "E", fontsize=15)

plt.tight_layout()
plt.savefig("./shap_summary_E.png", dpi=1200, format="png")
plt.close()


# 2. Identify the name or index of the feature you want to exclude
feature_to_exclude = "em"  # <-- Replace with the actual feature name

# 3. Get the index of this feature in the dataframe
#    If the feature is guaranteed to be in X.columns
idx_to_exclude = list(X.columns).index(feature_to_exclude)

# 4. Create new arrays that exclude that feature's column
shap_values_excl = shap_values[:, [i for i in range(shap_values.shape[1]) if i != idx_to_exclude]]
X_train_excl = X_train.iloc[:, [i for i in range(X_train.shape[1]) if i != idx_to_exclude]]

# 5. Plot a new SHAP summary plot using the reduced SHAP arrays
shap.summary_plot(
    shap_values_excl,
    X_train_excl,
    feature_names=[col for col in X.columns if col != feature_to_exclude],  # or X_train_excl.columns
    show=False
)

# 6. Optionally style the plot as you wish
fig, ax = plt.gcf(), plt.gca()
fig.set_size_inches(6, 4)

ax.set_xlabel("SHAP value", fontsize=15)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Adjust the size of the colorbar label and tick labels
fig.axes[-1].yaxis.label.set_size(15)
fig.axes[-1].set_yticklabels([tick.get_text() for tick in fig.axes[-1].get_yticklabels()], fontsize=15)

# 7. Add a label in the bottom corner (just as an example)
plt.text(28, 0.1, "E", fontsize=15)

plt.tight_layout()

# 8. Save the new plot
plt.savefig("./shap_summary_E_noTopFeature.png", dpi=1200, format="png")
plt.close()
