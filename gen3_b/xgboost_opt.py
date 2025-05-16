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
from pymatviz import density_hexbin


random.seed(42)

# Load the data
data = pd.read_csv("data/fe_merged.csv")

# Separate features and target variable
X = data.drop(columns=["fe"])
y = data["fe"]

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

# Create parity plot
fig, ax = plt.subplots(figsize=(8, 6))
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_optimized})

ax = density_hexbin(
    x="Actual",
    y="Predicted",
    df=df,
    ax=ax,
    xlabel=r"Actual EF",
    ylabel=r"Predicted EF",
    best_fit_line=False,
    gridsize=40,
)
ax.set_aspect("auto")
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig("xgboost_parity.png")
plt.close()

# shapley values
explainer = shap.TreeExplainer(optimized_xgb_model)
shap_values = explainer.shap_values(X_scaled)

X_scaled_renamed = X_scaled.rename(
    columns=lambda fn: fn.replace("em", "EM")
)

print(X_scaled_renamed.columns)

# SHAP plot 1: Create a new SHAP summary plot with selected features
expl = shap.Explanation(values=shap_values,
                        base_values=shap_values.mean(0),
                        data=X_scaled.values,
                        feature_names=['Separation \n distance', 'Spectrum \n overlap', 'Sca/Abs \n Ratio', 'EM'])

# now call beeswarm with dot_size
shap.plots.beeswarm(
    expl,
    show=False,
)

fig, ax = plt.gcf(), plt.gca()
fig.set_size_inches(10, 6)
ax.set_xlabel("SHAP value", fontsize=18)
ax.spines["right"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_linewidth(1.5)
ax.spines["top"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
ax.spines["left"].set_linewidth(1.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
fig.axes[-1].yaxis.label.set_size(18)
fig.axes[-1].get_yticklabels()[0].set_fontsize(18)
fig.axes[-1].get_yticklabels()[-1].set_fontsize(18)

for label in ax.get_yticklabels():
    label.set_fontsize(18)

plt.savefig("./shap_summary.png", dpi=600, format="png")
plt.close()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(X_scaled.corr(), dtype=bool))

# Round correlation values to 2 decimal places and set very small values to 0
corr_matrix = X_scaled.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.01] = 0

# Plot the correlation matrix with the mask
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", mask=mask, fmt=".2f")
plt.savefig("./corr_matrix.png", dpi=600, format="png")
plt.close()
