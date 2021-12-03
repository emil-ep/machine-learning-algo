import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('3.classification/logistic_regression/Social_Network_Ads.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature scaling is not necessary for logistic regression, however still applying it will improve the final predictions
#do this only after splitting train and test, becuase to avoid information leakage from test set
sc = StandardScaler()
print(X_test)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

#prediction of a single item , the first one in X_test
single_item = [[30, 87000]]
single_item_pred = classifier.predict(sc.fit_transform(single_item))
print(single_item_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)