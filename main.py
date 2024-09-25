import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle




# Load dataset
dataset = pd.read_csv('boston.csv')
dataset.head()
dataset.describe()

# Check and handle missing values
dataset.isnull().sum()
dataset = dataset.dropna(how='any').copy()
dataset.isnull().sum()


# corr_matrix = dataset.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.savefig('correlation_matrix.png')

# Preparing data
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

# Splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
pickle.dump(scaler, open('scaling.pkl', 'wb'))

# Model training
regression = LinearRegression()
regression.fit(X_train, Y_train)
print(regression.coef_)
print(regression.intercept_)
regression.get_params()

# Predicting
reg_predict = regression.predict(X_test)

# Calculating residuals
residual = Y_test - reg_predict

# Model evaluation metrics
print(mean_squared_error(Y_test, reg_predict))
print(mean_absolute_error(Y_test, reg_predict))
print(np.sqrt(mean_squared_error(Y_test, reg_predict)))

# R-squared and adjusted R-squared calculation
score = r2_score(Y_test, reg_predict)
print(score)
adjusted_r_squared = 1 - (1 - score) * (len(Y_test) - 1) / (len(Y_test) - X_test.shape[1] - 1)
print(adjusted_r_squared)

# Save the model
pickle.dump(regression, open('regmodel.pkl', 'wb'))
