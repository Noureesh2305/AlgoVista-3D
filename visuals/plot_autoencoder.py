# visuals/plot_autoencoder.py

import streamlit as st
import plotly.subplots as sp
import plotly.graph_objects as go
from models.autoencoder import AutoencoderModel
from sklearn.datasets import load_digits

def render():
    st.header("🧠 Autoencoder – Dimensionality Reduction & Reconstruction")

    st.markdown("""
    An **Autoencoder** is a type of neural network that learns to **compress and reconstruct** data.  
    It's commonly used for **dimensionality reduction**, **denoising**, and **anomaly detection**.
    
    - 🔐 Encoder compresses input into lower dimensions  
    - 🔄 Decoder reconstructs the original input  
    - 🧠 Learns unsupervised representations
    """)

    # Experiment Mode: Hidden Layer Size
    hidden_units = st.slider("🎛️ Hidden Layer Size (Controls Compression)", min_value=4, max_value=128, value=32, step=4)
    model = AutoencoderModel(hidden_dim=hidden_units)
    original, reconstructed = model.get_original_reconstructed()
    digits = load_digits()

    # Plot 5 images: Original vs Reconstructed
    fig = sp.make_subplots(rows=2, cols=5, subplot_titles=[f"Digit {i}" for i in range(5)])

    for i in range(5):
        orig_img = original[i].reshape(8, 8)
        recon_img = reconstructed[i].reshape(8, 8)

        fig.add_trace(go.Heatmap(z=orig_img, showscale=False, colorscale='Viridis'), row=1, col=i+1)
        fig.add_trace(go.Heatmap(z=recon_img, showscale=False, colorscale='Viridis'), row=2, col=i+1)

    fig.update_layout(
        height=600,
        title_text="Autoencoder Reconstruction – Top: Original | Bottom: Reconstructed",
        showlegend=False,
        margin=dict(t=50, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("✅ Try changing the hidden layer size above to observe how reconstruction improves or degrades.")
