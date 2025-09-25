from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def run_kmeans_library(X, n_clusters):
    """
    Executa o K-means usando a biblioteca Scikit-learn.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    score = silhouette_score(X, labels)
    print(f"K-means com Biblioteca (K={n_clusters}) - Silhouette Score: {score:.4f}")
    return labels, centroids, score