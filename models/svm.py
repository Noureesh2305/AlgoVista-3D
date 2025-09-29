# models/svm.py

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def train_svm_model():
    X, y = make_classification(n_samples=200, n_features=3, n_informative=3,
                               n_redundant=0, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return X_train, X_test, y_train, y_pred, model
