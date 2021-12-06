import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('3.classification/kernal_support_vector/Social_Network_Ads.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
sc = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#using RBF gaussian kernel instead of linear
classifier = SVC(kernel = 'rbf', random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

single_item = classifier.predict(sc.fit_transform([[30, 87000]]))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)