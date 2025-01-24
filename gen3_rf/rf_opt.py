import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Fix all the random seeds
import random
random.seed(42)

# Load the data
data = pd.read_csv("data/fe_merged.csv")

# Separate features and target variable
X = data[["distance", "abs", "sca", "plasmon"]]
y = data["fe"]

# Change the column name abs to absorption, sca to scattering, and plasmon to plasmon peak
X = X.rename(columns={"abs": "absorption", "sca": "scattering", "plasmon": "plasmon peak"})

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up the grid search
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
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
optimized_rf_model = RandomForestRegressor(**best_params, random_state=42)
optimized_rf_model.fit(X_train, y_train)

# Predicting and evaluating the model with optimized parameters
y_pred_optimized = optimized_rf_model.predict(X_test)
r2_optimized = r2_score(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)

print(f"R2 on test set for E: {r2_optimized:.4f}, MSE: {mse_optimized:.4f}")

# Plot the parity plot
sns.set()
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_optimized, s=150)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "k--")
plt.xlabel("Actual E", fontsize=32)
plt.ylabel("Predicted E", fontsize=32)
plt.title("Single-Target Random Forest Prediction for E\n$R^2$: {:.3f}, MSE: {:.3f}".format(r2_optimized, mse_optimized), fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("./parity_plot_E_rf.svg", dpi=1200, format="svg")
plt.close()

# SHAP values
explainer = shap.TreeExplainer(optimized_rf_model)
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

plt.tight_layout()
plt.savefig("./shap_summary_E_rf.png", dpi=600, format="png")
plt.close()

# Correlation matrix heatmap
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.savefig("./corr_matrix_rf.png", dpi=600, format="png")
plt.close()
