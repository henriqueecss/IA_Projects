from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def run_kmeans_library(X, n_clusters):
    """
    Executa o K-means usando a biblioteca Scikit-learn.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    score = silhouette_score(X, labels)
    return labels, centroids, score

def perform_pca(X, n_components):
    """
    Aplica a técnica de PCA para redução de dimensionalidade.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca.components_