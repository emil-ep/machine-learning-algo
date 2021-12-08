import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('5.clustering/k-means-clustering/Mall_Customers.csv')
#we are only taking 3 and 4 columns here because of visualising purposes. we can actually use all features except the 
#first column which is customer Id
X = dataset.iloc[:, [3, 4]].values
y = dataset.iloc[:, -1].values



