import streamlit as st
import plotly.express as px
import numpy as np
from tutor.voice_guide import speak_theory

def render():
    # Generate dummy data for demo purposes
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    # Create sample data
    X, _ = make_blobs(n_samples=300, centers=4, n_features=3, random_state=42)

    # Train KMeans
    model = KMeans(n_clusters=4)
    model.fit(X)
    labels = model.labels_

    # Plotting
    fig = px.scatter_3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        color=labels.astype(str),
        title="K-Means Clustering – 3D View",
        labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3'}
    )
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True)

    # Theory section
    st.markdown("### 📘 K-Means Clustering – Theory")
    theory = """
    **K-Means Clustering** is an **unsupervised learning algorithm** used to group similar data points into clusters.
    
    - It tries to partition data into **K distinct clusters** based on distance to the centroid.
    - The algorithm:
        1. Randomly selects K centroids.
        2. Assigns each point to the nearest centroid (forming clusters).
        3. Updates centroids as the mean of assigned points.
        4. Repeats until convergence (no change in centroids or max iterations).
    - **Objective**: Minimize the within-cluster sum of squares (WCSS).
    
    💡 Use Case Examples: Customer segmentation, Image compression, Document clustering.
    """
    st.markdown(theory)

    if st.button("🔊 Play Voice Explanation"):
        speak_theory(theory)
