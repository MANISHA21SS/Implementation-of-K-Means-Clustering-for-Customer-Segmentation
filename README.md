# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1 : Data Preparation:
1.1 Collect customer data (e.g., age, income, purchase history).
1.2 Preprocess the data:
- Clean the data (remove duplicates, handle missing values).
- Normalize numerical attributes (e.g., using StandardScaler).

Step 2 : Determine the Number of Clusters (K):
2.1 Use techniques like the Elbow Method or Silhouette Score to identify the optimal value of K.
2.2 Plot WCSS (Within-Cluster Sum of Squares) vs. number of clusters to find the "elbow" point.

Step 3 : Initialize Centroids:
3.1 Randomly select K data points from the dataset as the initial centroids.

Step 4 : Assign Points to Clusters:
4.1 For each data point, calculate the distance to each centroid (e.g., using Euclidean distance).
4.2 Assign the data point to the cluster with the nearest centroid.

Step 5 : Update Centroids:
5.1 Calculate the new centroid of each cluster as the mean of all points in that cluster.

Step 6 : Iteration:
6.1 Repeat steps 4 and 5 until convergence:
- Convergence occurs when centroids no longer change or the maximum number of iterations is reached.

Step 7: Evaluate Clustering:
7.1 Calculate evaluation metrics (WCSS, Silhouette Score).
7.2 Visualize clusters (e.g., scatter plot with different colors for each cluster).

Step 8 : Analysis and Interpretation:
8.1 Analyze cluster characteristics and derive customer segments.
8.2 Label clusters with meaningful names based on their characteristics (e.g., "High Spenders").

Step 9 : Deployment:
9.1 Integrate the clustering model into a system for real-time customer segmentation.
9.2 Continuously update the model as new data becomes available. 

## Program:
/*
Program to implement the K Means Clustering for Customer Segmentation.

Developed by: Manisha selvakumari.S.S.

RegisterNumber:  212223220055
*/
```
print("Name : MANISHA SELVAKUMARI S S")
print("Reg No : 212223220055")

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Mall_Customers.csv')

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++")
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])

KMeans(n_clusters = 5)

y_pred = km.predict(data.iloc[:, 3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```
## Output:
![Screenshot (264)](https://github.com/user-attachments/assets/cfc64a29-cf28-439d-98a8-c8645c7c6e4d)

![Screenshot 2025-05-19 162208](https://github.com/user-attachments/assets/fee4c1dc-f347-46a3-8d6a-c79a73dd1764)

![Screenshot 2025-05-19 162221](https://github.com/user-attachments/assets/3f06fd2b-2f61-4fc1-827a-f1cafa87e91b)

![Screenshot 2025-05-19 162233](https://github.com/user-attachments/assets/358091d2-6d7c-45ec-8118-96ecd214ff8d)

![Screenshot (265)](https://github.com/user-attachments/assets/8f5fbd26-55c6-4321-91cb-6cb2be9b5def)

![Screenshot (266)](https://github.com/user-attachments/assets/6364f9b3-00d0-466d-b28b-c60b8ea6597e)

![Screenshot (267)](https://github.com/user-attachments/assets/c094a0ed-d7dd-4b0c-b5aa-a0117390348c)

![Screenshot 2025-05-19 162341](https://github.com/user-attachments/assets/62d1b936-ab9b-4d9b-abea-4089eec1783a)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
