# models/hierarchical.py

from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

def train_hierarchical_model(n_samples=50, n_clusters=3, method='ward'):
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=1.0, random_state=42)
    linked = linkage(X, method=method)
    labels = fcluster(linked, t=n_clusters, criterion='maxclust')
    return X, linked, labels
