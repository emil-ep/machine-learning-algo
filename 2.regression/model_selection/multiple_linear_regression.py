from matplotlib.pyplot import sca
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv("2.regression/model_selection/Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#we are using the same class for simple linear regression here because it will take of the multiple features
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

score = r2_score(y_test, y_pred)
print(score)