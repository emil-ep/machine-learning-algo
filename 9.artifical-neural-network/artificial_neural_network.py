import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.sputils import matrix
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


print(tf.__version__)

dataset = pd.read_csv('9.artifical-neural-network/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
#Label encoding of the gender column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#splitting the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#FEATURE SCALING IS MANDATORY FOR DEEP LEARNING
#IT IS MANDATORY TO APPLY FEATURE SCALING FOR ALL THE FEATURES
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building the artificial neural network
ann = tf.keras.models.Sequential()
#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#units indicate the number of neurons
#how do we know how many neurons we need, it is found out by experimentaiton and no rule of thumb. Usually by 
#considering the hyperparameters
#Adding the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#Compiling the artificial neural network
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the artificial neural network
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Predicting a single prediction for the following observation
#greography - france (France corresponds to 1, 0, 0 because of onehotEncoding)
#credit score - 600
#Gender - male (1 for male and 0 for female)
#age - 40 years old
#Tenure - 3 years
#Balance - 60000$
#Number of products - 2
#Does this customer has a credit card - Yes
#Is this customer an active member - yes
#Estimated salary - 50000$

#we need to provide the input in the scaled value (we applied feature scaling while providing input for training)
output = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print(output)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


