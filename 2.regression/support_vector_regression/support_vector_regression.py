import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('2.regression/support_vector_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#if we are using feature scaling, we need to inverse feature scale so as to get the original value
# print(X)
# print (y)
# y is a singular araay. But it wont work because standard scaler expects a 2D array
y = y.reshape(len(y), 1)
# print(y)

#feature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
#we need to input 2 standard scaler becuase the independent mean and avergae will be stored in sc object
sc_y =  StandardScaler()
y = sc_y.fit_transform(y)

# print(X)
# print(y)
#RBF kernel is used for non linear dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
#we feature scaled the input dataset, so for prediction also, we need to feature scale the input
y_pred = regressor.predict(sc_X.transform([[6.5]]))
print(sc_y.inverse_transform(y_pred))