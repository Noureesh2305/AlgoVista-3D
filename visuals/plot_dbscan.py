# visuals/plot_dbscan.py

import streamlit as st
import plotly.express as px
from models.dbscan import train_dbscan_model

def render():
    st.markdown("## 🌌 DBSCAN Clustering")
    st.markdown("""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering method that finds **core samples** of high density and expands clusters from them.

    - Good for **non-linear cluster shapes**
    - Handles **noise/outliers**

    Parameters:
    - `eps`: Maximum distance between points
    - `min_samples`: Minimum points required to form a dense region
    """)

    X, labels = train_dbscan_model()
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels.astype(str),
                     title="DBSCAN Clustering", labels={'color': 'Cluster'})
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig)
    st.success("✅ DBSCAN identifies clusters and outliers without predefining the number of clusters.")
