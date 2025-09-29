# experiment_mode/dbscan_experiment.py

import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def render():
    st.subheader("🔍 DBSCAN – Density-Based Clustering (Experiment Mode)")

    st.markdown("""
    **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is a clustering algorithm that groups together points that are close to each other (density) and marks points that lie alone in low-density regions as outliers.
    
    - Does not require number of clusters as input.
    - Can find arbitrarily shaped clusters.
    - Robust to noise and outliers.
    """)

    # DBSCAN Parameters
    eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=1.0, step=0.1, value=0.3)
    min_samples = st.slider("Min Samples", min_value=2, max_value=10, step=1, value=5)

    # Generate synthetic dataset
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    # Plotting
    df = {
        "x": X_scaled[:, 0],
        "y": X_scaled[:, 1],
        "cluster": labels.astype(str)
    }

    fig = px.scatter_3d(
        df, x="x", y="y", z=[0]*len(X_scaled), color="cluster",
        title="DBSCAN Cluster Visualization",
        labels={"cluster": "Cluster"},
        height=500
    )

    st.plotly_chart(fig)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    st.markdown(f"""
    - **Estimated Clusters**: `{n_clusters}`
    - **Noise Points**: `{n_noise}`
    """)
