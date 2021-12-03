import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


dataset = pd.read_csv("2.regression/polynomial_linear_regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# we are not using splitting dataset into train and test set because we have small data
#we are creating both linear regression model and polynomial regressio model to compare
# the difference
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#polynomial linear regression model is a combination of power of the linear features
poly_reg = PolynomialFeatures(degree=2)
#This is a matrix of features with polynomial degree of 2
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#polynomial linear regression model is a combination of power of the linear features
poly_reg_4 = PolynomialFeatures(degree=4)
#This is a matrix of features with polynomial degree of 4
X_poly_4 = poly_reg_4.fit_transform(X)

lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly_4, y)


#visualising the data for linear regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial regression - degree 2)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_4.predict(X_poly_4), color = 'blue')
plt.title('Truth or Bluff (Polynomial regression - degree 4)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#since it is a degree of 4, we can't input just [[6.5]], it needs to be transformed into 6.5, 6.5^2, 6.5^3, 6.5^4
#that is what we are doing with the poly_reg_4.fit_transform method
output = lin_reg_4.predict(poly_reg_4.fit_transform([[6.5]]))
print(output)