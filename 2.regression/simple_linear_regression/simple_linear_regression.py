import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("2.regression/simple_linear_regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("**** X_test *****")
print(X_test)
print("**** y_test *****")
print(y_test)
y_pred = regressor.predict(X_test)
print("**** y_pred *****")
print(y_pred)

#plotting the result using matplotlib
plt.scatter(X_train, y_train, color = 'red')
#plotting the X_train predictions
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
#plotting the X_test predictions
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


#Bonus - predict the salary of a single years of experience
#For eg: predict the salary of an employee with 12 years of experience
salary_of_12_year_exp = regressor.predict([[12]])
print(salary_of_12_year_exp)

#Bonus - get the coefficients and intercept of the final regression equation
# y = b0 + b1*x1
print(regressor.coef_)
print(regressor.intercept_)
#Salary = regressor.coef_ * 12 + regressor.intercept_