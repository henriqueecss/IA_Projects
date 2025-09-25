import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import time

from kmeans_hardcore import KMeansHardcore
from kmeans_library import run_kmeans_library

def load_data():
    """
    Carrega a base de dados Iris.
    """
    iris = load_iris()
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names).to_numpy()
    return X, iris.target

def perform_pca_and_plot(X, labels, centroids, n_components, title):
    """
    Aplica PCA e plota os resultados da clusterização.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    if n_components == 2:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
        plt.scatter(pca.transform(centroids)[:, 0], pca.transform(centroids)[:, 1], marker='X', s=200, color='red', label='Centróides')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
    elif n_components == 1:
        plt.scatter(X_reduced, np.zeros_like(X_reduced), c=labels, cmap='viridis')
        plt.scatter(pca.transform(centroids), np.zeros_like(pca.transform(centroids)), marker='X', s=200, color='red', label='Centróides')
        plt.xlabel('Componente Principal 1')
        plt.yticks([])
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    X, y_true = load_data()

    # --- Experimento com K=3 ---
    print("--- Executando Experimento K=3 ---")
    
    # K-means Hardcore
    start_time_hc_k3 = time.time()
    kmeans_hc_k3 = KMeansHardcore(n_clusters=3)
    labels_hc_k3, centroids_hc_k3 = kmeans_hc_k3.fit(X, initial_centroids=[0, 50, 100])
    end_time_hc_k3 = time.time()
    score_hc_k3 = silhouette_score(X, labels_hc_k3)
    print(f"K-means Hardcore (K=3) - Tempo de execução: {end_time_hc_k3 - start_time_hc_k3:.4f}s | Silhouette Score: {score_hc_k3:.4f}\n")

    # K-means com Biblioteca
    start_time_lib_k3 = time.time()
    labels_lib_k3, centroids_lib_k3, score_lib_k3 = run_kmeans_library(X, 3)
    end_time_lib_k3 = time.time()
    print(f"K-means com Biblioteca (K=3) - Tempo de execução: {end_time_lib_k3 - start_time_lib_k3:.4f}s\n")
    
    # --- Experimento com K=5 ---
    print("--- Executando Experimento K=5 ---")

    # K-means Hardcore
    start_time_hc_k5 = time.time()
    kmeans_hc_k5 = KMeansHardcore(n_clusters=5)
    labels_hc_k5, centroids_hc_k5 = kmeans_hc_k5.fit(X)
    end_time_hc_k5 = time.time()
    score_hc_k5 = silhouette_score(X, labels_hc_k5)
    print(f"K-means Hardcore (K=5) - Tempo de execução: {end_time_hc_k5 - start_time_hc_k5:.4f}s | Silhouette Score: {score_hc_k5:.4f}\n")

    # K-means com Biblioteca
    start_time_lib_k5 = time.time()
    labels_lib_k5, centroids_lib_k5, score_lib_k5 = run_kmeans_library(X, 5)
    end_time_lib_k5 = time.time()
    print(f"K-means com Biblioteca (K=5) - Tempo de execução: {end_time_lib_k5 - start_time_lib_k5:.4f}s\n")

    # --- Análise e Visualização (para o Relatório) ---
    print("--- Gerando visualizações com PCA ---")
    best_k = 3
    
    # Plota os resultados da implementação Hardcore (melhor K) com PCA
    perform_pca_and_plot(X, labels_hc_k3, centroids_hc_k3, 2, 'K-means Hardcore (K=3) com PCA 2D')
    perform_pca_and_plot(X, labels_hc_k3, centroids_hc_k3, 1, 'K-means Hardcore (K=3) com PCA 1D')

    # Plota os resultados da implementação com Biblioteca (melhor K) com PCA
    perform_pca_and_plot(X, labels_lib_k3, centroids_lib_k3, 2, 'K-means Biblioteca (K=3) com PCA 2D')
    perform_pca_and_plot(X, labels_lib_k3, centroids_lib_k3, 1, 'K-means Biblioteca (K=3) com PCA 1D')

if __name__ == "__main__":
    main()