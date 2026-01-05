import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
import time
import numpy as np

from kmeans_hardcore import KMeansHardcore
from kmeans_library import run_kmeans_library, perform_pca

def load_data():
    """
    Carrega a base de dados Iris.
    """
    iris = load_iris()
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names).to_numpy()
    y_true = iris.target
    return X, y_true

def plot_clusters(X_reduced, labels, centroids, n_components, title):
    """
    Plota os resultados da clusterização após a redução de dimensionalidade com PCA.
    """
    plt.figure(figsize=(8, 6))
    if n_components == 2:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red', label='Centróides')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
    elif n_components == 1:
        plt.scatter(X_reduced, np.zeros_like(X_reduced), c=labels, cmap='viridis')
        plt.scatter(centroids, np.zeros_like(centroids), marker='X', s=200, color='red', label='Centróides')
        plt.xlabel('Componente Principal 1')
        plt.yticks([])
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def map_labels_to_classes(y_pred, y_true):
    """
    Mapeia os rótulos de cluster para os rótulos de classe reais
    e retorna o resultado do mapeamento e as métricas de acurácia.
    """
    from scipy.stats import mode
    labels = np.unique(y_pred)
    mappings = {}
    for label in labels:
        mask = (y_pred == label)
        mapped_class = mode(y_true[mask])[0]
        mappings[label] = mapped_class
    
    y_mapped = np.array([mappings[label] for label in y_pred])
    
    return y_mapped

def main():
    X, y_true = load_data()

    # --- Experimento com K=3 ---
    print("--- Executando Experimento K=3 ---")
    
    # K-means Hardcore
    start_time_hc_k3 = time.time()
    kmeans_hc_k3 = KMeansHardcore(n_clusters=3)
    labels_hc_k3, centroids_hc_k3 = kmeans_hc_k3.fit(X, initial_centroids=[0, 50, 100])
    end_time_hc_k3 = time.time()
    
    # K-means com Biblioteca
    start_time_lib_k3 = time.time()
    labels_lib_k3, centroids_lib_k3, score_lib_k3 = run_kmeans_library(X, 3)
    end_time_lib_k3 = time.time()
    
    # --- Análise e Comparação de Métricas (K=3) ---
    score_hc_k3 = silhouette_score(X, labels_hc_k3)
    print(f"K-means Hardcore (K=3) - Tempo: {end_time_hc_k3 - start_time_hc_k3:.4f}s | Silhouette: {score_hc_k3:.4f}")
    print(f"K-means com Biblioteca (K=3) - Tempo: {end_time_lib_k3 - start_time_lib_k3:.4f}s | Silhouette: {score_lib_k3:.4f}\n")

    # --- Experimento com K=5 ---
    print("--- Executando Experimento K=5 ---")

    # K-means Hardcore
    start_time_hc_k5 = time.time()
    kmeans_hc_k5 = KMeansHardcore(n_clusters=5)
    labels_hc_k5, centroids_hc_k5 = kmeans_hc_k5.fit(X)
    end_time_hc_k5 = time.time()
    
    # K-means com Biblioteca
    start_time_lib_k5 = time.time()
    labels_lib_k5, centroids_lib_k5, score_lib_k5 = run_kmeans_library(X, 5)
    end_time_lib_k5 = time.time()

    # --- Análise e Comparação de Métricas (K=5) ---
    score_hc_k5 = silhouette_score(X, labels_hc_k5)
    print(f"K-means Hardcore (K=5) - Tempo: {end_time_hc_k5 - start_time_hc_k5:.4f}s | Silhouette: {score_hc_k5:.4f}")
    print(f"K-means com Biblioteca (K=5) - Tempo: {end_time_lib_k5 - start_time_lib_k5:.4f}s | Silhouette: {score_lib_k5:.4f}\n")

    # --- Avaliação Adicional (opcional para relatório) ---
    # Comparando os rótulos de cluster com os rótulos de classe verdadeiros
    best_k = 3
    labels_hc_best = labels_hc_k3
    labels_lib_best = labels_lib_k3
    
    mapped_hc_labels = map_labels_to_classes(labels_hc_best, y_true)
    mapped_lib_labels = map_labels_to_classes(labels_lib_best, y_true)
    
    print("--- Avaliação com classes originais ---")
    print(f"Homogeneidade Hardcore: {homogeneity_score(y_true, mapped_hc_labels):.4f}")
    print(f"Homogeneidade Biblioteca: {homogeneity_score(y_true, mapped_lib_labels):.4f}")
    print(f"Completude Hardcore: {completeness_score(y_true, mapped_hc_labels):.4f}")
    print(f"Completude Biblioteca: {completeness_score(y_true, mapped_lib_labels):.4f}\n")

    # --- Gerando visualizações com PCA ---
    print("--- Gerando visualizações com PCA ---")
    
    # K-means Hardcore (melhor k=3)
    X_reduced_2d_hc, _ = perform_pca(X, 2)
    labels_hc_pca_2d, centroids_hc_pca_2d = kmeans_hc_k3.fit(X_reduced_2d_hc)
    plot_clusters(X_reduced_2d_hc, labels_hc_pca_2d, centroids_hc_pca_2d, 2, 'K-means Hardcore (K=3) com PCA 2D')

    X_reduced_1d_hc, _ = perform_pca(X, 1)
    labels_hc_pca_1d, centroids_hc_pca_1d = kmeans_hc_k3.fit(X_reduced_1d_hc)
    plot_clusters(X_reduced_1d_hc, labels_hc_pca_1d, centroids_hc_pca_1d, 1, 'K-means Hardcore (K=3) com PCA 1D')

    # K-means Biblioteca (melhor k=3)
    X_reduced_2d_lib, _ = perform_pca(X, 2)
    labels_lib_pca_2d, centroids_lib_pca_2d, _ = run_kmeans_library(X_reduced_2d_lib, 3)
    plot_clusters(X_reduced_2d_lib, labels_lib_pca_2d, centroids_lib_pca_2d, 2, 'K-means Biblioteca (K=3) com PCA 2D')

    X_reduced_1d_lib, _ = perform_pca(X, 1)
    labels_lib_pca_1d, centroids_lib_pca_1d, _ = run_kmeans_library(X_reduced_1d_lib, 3)
    plot_clusters(X_reduced_1d_lib, labels_lib_pca_1d, centroids_lib_pca_1d, 1, 'K-means Biblioteca (K=3) com PCA 1D')

if __name__ == "__main__":
    main()