from sklearn.datasets import make_classification, make_blobs, make_regression
import numpy as np

def generate_knn_data():
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=4)
    return X, y

def generate_svm_data():
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=7)
    return X, y

def generate_logistic_regression_data():
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=1)
    return X, y

def generate_linear_regression_data():
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
    return X, y

def generate_decision_tree_data():
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=3)
    return X, y

def generate_random_forest_data():
    X, y = make_classification(n_samples=150, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=42)
    return X, y

def generate_naive_bayes_data():
    X, y = make_classification(n_samples=120, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=5)
    return X, y

def generate_kmeans_data():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)
    return X, y

def generate_pca_data():
    X, y = make_classification(n_samples=120, n_features=5, n_redundant=0,
                               n_informative=3, n_clusters_per_class=1, random_state=10)
    return X, y
