import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


dataset = pd.read_csv('2.regression/decision_tree/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#we don't require to include feature scaling for decision tree regression
#don't add much parameters 
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])
print(y_pred)

#decision tree model is not optimised for single colomn dataset, there should be many features. Thats the best case
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) 
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue' )
plt.title('Truth or Bluff (Decision tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()