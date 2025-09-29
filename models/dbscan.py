# models/dbscan.py

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

def train_dbscan_model():
    from sklearn.preprocessing import StandardScaler
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    model = DBSCAN(eps=0.3, min_samples=5).fit(X)
    labels = model.labels_
    return X, labels
