import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from Best_Components_PCA import pca_result, data
from sklearn.metrics import silhouette_score


eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = range(3,15)

best_eps = None
best_min_samples = None
best_score = -1

#Grid search for best parameters for the model
for eps in eps_values:
    for sam_val in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=sam_val)
        labels = dbscan.fit_predict(pca_result)

        if len(np.unique(labels)) > 1 and -1 in labels:
            score = silhouette_score(pca_result,labels)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = sam_val

dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_clusters = dbscan.fit_predict(pca_result)

data['DBSCAN_Cluster'] = dbscan_clusters
print(f"Best Epsilon {best_eps}, Best Mini Samples {best_min_samples}")

#Chat GPT CODE
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    pca_result[:, 2],
    c=dbscan_clusters,
    cmap='viridis',
    s=20
)
ax.set_title("DBSCAN Clustering with PCA (3D)")
fig.colorbar(scatter, ax=ax)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
plt.show()

plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_clusters, cmap='viridis', s=20)
plt.title("DBSCAN Clustering with PCA (2D)")
plt.colorbar()
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
### END CHATGPT  CODE
