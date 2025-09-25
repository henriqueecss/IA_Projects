import numpy as np

class KMeansHardcore:
    """
    Implementação 'hardcore' do algoritmo K-means.
    """
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X, initial_centroids=None):
        if initial_centroids is not None:
            self.centroids = X[initial_centroids]
        else:
            np.random.seed(42)
            random_indices = np.random.permutation(X.shape[0])
            self.centroids = X[random_indices[:self.n_clusters]]

        for i in range(self.max_iter):
            # Passo 1: Atribuir cada ponto ao centróide mais próximo
            self.labels = self._assign_clusters(X)
            
            # Passo 2: Recalcular os centróides
            new_centroids = self._update_centroids(X)

            # Condição de parada: se os centróides não mudaram, pare.
            if np.allclose(self.centroids, new_centroids):
                print(f"K-means Hardcore - Convergiu na iteração {i+1}.")
                break
            
            self.centroids = new_centroids

        return self.labels, self.centroids

    def _assign_clusters(self, X):
        labels = np.zeros(X.shape[0], dtype=int)
        for i, point in enumerate(X):
            distances = np.linalg.norm(point - self.centroids, axis=1)
            labels[i] = np.argmin(distances)
        return labels

    def _update_centroids(self, X):
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            points_in_cluster = X[self.labels == k]
            if len(points_in_cluster) > 0:
                new_centroids[k] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[k] = self.centroids[k]
        return new_centroids