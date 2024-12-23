from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('ml-pillar.csv')

# Separate features and target variable
X = data.drop(['wavelength', 'E', 'Q'], axis=1)
y = data['E']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=539)

# Create the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

mse = mean_squared_error(y_test, predicted)
r2 = regressor.score(X_test, y_test)
print('MSE on test set: ', mse)
print('R2 on test set: ', r2)


