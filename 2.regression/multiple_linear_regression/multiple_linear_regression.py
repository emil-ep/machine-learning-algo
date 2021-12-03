from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("2.regression/multiple_linear_regression/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#we are using the same class for simple linear regression here because it will take of the multiple features
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.printoptions(precision=2)
# print("***** y_test *****")
# print(y_test)
# print("***** y_pred *****")
# print(y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

#Making a single prediction (for example the profit of a startup with R&D 
# Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
