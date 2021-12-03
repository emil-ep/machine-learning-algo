import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('/Users/emil/Documents/study-topics/machine-learning/udemy/machine_learrning_a-z_project/1.data-preprocessing/Data.csv')
X = dataset.iloc[:, :-1].values #all columns except the last column
y = dataset.iloc[:, -1].values #all only the last column
print(X)
# replace all missing values with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#include the missing values in the age and salary column from dataset
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
#one hot encode the country column - one hot encoding is used because we don't want to provide a relationship between the order of these countries
#so instead, machine learning will interpret each of the country as an independent vector
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
#we will encode the 'purchased' column with Label encoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
#Question - Do we need to apply feature scaling to the dataset before splitting it into training and  test set?
#Ans - We have to apply feature scaling after we split the dataset into train and test set
#Feature scaling is applied to ignore if the training set contains feature values which will dominate the other, just to keep everything
#under the same range , feature scaling will ignore such values which will dominate the other values

#splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("**** X train ****")
print(X_train)
print("**** X test ****")
print(X_test)
print("**** y train ****")
print(y_train)
print("**** y test ****")
print(y_test)

#Feature scaling
#There are two feature scaling techniques
#1. Standardisation = (x - mean(x))/ standard_deviation(x) - all output comes in the range [-3, 3]
#2. Normalisation = (x - min(x))/max(x) - min(x) - all output comes in the range [0, 1]
#Normalisation is recommended when there is a normalised data
#Standardisation works most of the time

#Do we need to apply feature scaling to the dummy variables created in one hot encoding?
#Ans - no, we don't need to
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
#Don't apply fit method for test dataset, since it will produce two scalars
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("****** standardised X_train *******")
print(X_train)
print("****** standardised X_test *******")
print(X_test)