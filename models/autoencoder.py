# models/autoencoder.py

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

class AutoencoderModel:
    def __init__(self, hidden_dim=32):
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(self.hidden_dim,), 
                                  max_iter=500, random_state=42)

    def load_and_train(self):
        # Load and scale data
        data = load_digits()
        X = data.data
        X_scaled = self.scaler.fit_transform(X)

        # Split and train
        X_train, _, y_train, _ = train_test_split(X_scaled, X_scaled, 
                                                  test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Return original and reconstructed data
        reconstructed = self.model.predict(X_scaled)
        return X_scaled, reconstructed

    def get_original_reconstructed(self):
        return self.load_and_train()
