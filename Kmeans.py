from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Best_Components_PCA import pca_result
import numpy as np


cluster_range = range(2, 11)
inertia_scores = []
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pca_result)
    inertia_scores.append(kmeans.inertia_)

    silhouette = silhouette_score(pca_result, labels)
    silhouette_scores.append(silhouette)

best_k = cluster_range[silhouette_scores.index(max(silhouette_scores))]
best_silhoutte_score = max(silhouette_scores)
print(best_k,best_silhoutte_score)

kmeans = KMeans(n_clusters=best_k, random_state=42)

#CHATGPT START CODE
kmeans_labels = kmeans.fit_predict(pca_result)


plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis', s=50)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')

plt.title("K-Means Clusters in 2D")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()
#CHATGPT END CODE
unique, counts = np.unique(kmeans_labels, return_counts=True)

cluster_sizes = dict(zip(unique, counts))
print("Cluster sizes:", cluster_sizes)
