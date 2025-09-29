# experiment_mode/autoencoder_experiment.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return autoencoder

def render():
    st.subheader("🧠 Autoencoder – Experiment Mode")

    st.markdown("""
    An **Autoencoder** is a neural network used to learn efficient codings of unlabeled data.
    
    - It learns to compress data (encoding) and then reconstruct it (decoding).
    - Commonly used in anomaly detection, image compression, and denoising.
    
    """)
    
    encoding_dim = st.slider("Encoding Dimension", min_value=2, max_value=32, value=16, step=2)

    digits = load_digits()
    data = digits.data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    autoencoder = build_autoencoder(input_dim=data.shape[1], encoding_dim=encoding_dim)
    history = autoencoder.fit(data_scaled, data_scaled, epochs=10, batch_size=64, verbose=0)

    # Encode and decode some digits
    encoded_imgs = autoencoder.predict(data_scaled)
    idx = st.slider("Sample Index", 0, len(data_scaled)-1, 0)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=np.arange(len(data_scaled[idx])),
        y=[0]*len(data_scaled[idx]),
        z=data_scaled[idx],
        mode='lines+markers',
        name='Original',
        line=dict(color='blue'),
        marker=dict(size=3)
    ))

    fig.add_trace(go.Scatter3d(
        x=np.arange(len(encoded_imgs[idx])),
        y=[1]*len(encoded_imgs[idx]),
        z=encoded_imgs[idx],
        mode='lines+markers',
        name='Reconstructed',
        line=dict(color='green'),
        marker=dict(size=3)
    ))

    fig.update_layout(
        title="Original vs Reconstructed Data",
        scene=dict(
            xaxis_title='Feature Index',
            yaxis_title='Layer',
            zaxis_title='Value'
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig)
