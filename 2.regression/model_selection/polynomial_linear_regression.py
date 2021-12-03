import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


dataset = pd.read_csv("2.regression/model_selection/Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#polynomial linear regression model is a combination of power of the linear features
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)

regressor = LinearRegression()
regressor.fit(X_poly, y_train)

y_pred = regressor.predict(poly_reg.transform(X_test))
np.printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
score = r2_score(y_test, y_pred)
print(score)