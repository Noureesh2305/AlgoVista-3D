# models/kmeans.py

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def train_kmeans_model():
    data = load_iris()
    X = data.data
    y = data.target  # Not used in clustering, but useful for comparing later

    # Train KMeans model
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X)
    y_pred = model.labels_

    # Split manually for visualization purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y_pred, test_size=0.3, random_state=42)

    return X_train, y_train, X_test, y_test, model
