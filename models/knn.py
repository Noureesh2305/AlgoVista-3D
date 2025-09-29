# models/knn.py

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def train_knn_model(n_neighbors=5):
    # Create synthetic classification dataset
    X, y = make_classification(n_samples=200, n_features=3, n_informative=3, 
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return X_train, X_test, y_train, y_pred, model
