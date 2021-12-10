import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('5.clustering/k-means-clustering/Mall_Customers.csv')
#we are only taking 3 and 4 columns here because of visualising purposes. we can actually use all features except the 
#first column which is customer Id
#For clustering we don't need the y values, becuase we are not taking any predictions, we are clustering based on the inputs only
X = dataset.iloc[:, [3, 4]].values
#we are going to run kmeans clustering from 1 cluster to 10 cluster
#we will compute WCSS (within cluster sum of squares)
#on X axis, we will have number of clusters and on y axis we will have the wcss value
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

#From the graph that we plotted, we can understand that from cluster 5, the slope decreases
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
kmeans.fit(X)
#creating a dependent variable, which will be a group of customer in this case
y_pred = kmeans.fit_predict(X)
print(y_pred)
#y_pred outputs the cluster information of each entry in the dataset. 0,1,2,3,4 - 5 clusters

#visualising cluster below
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'magenta', label = 'cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.title('Clusters of customers')
plt.show()