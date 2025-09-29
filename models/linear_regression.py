# models/linear_regression.py

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_linear_regression_model():
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return X_train, X_test, y_train, y_pred, model
