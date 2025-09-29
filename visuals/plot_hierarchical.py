# visuals/plot_hierarchical.py

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from models.hierarchical import train_hierarchical_model
import pandas as pd

def render():
    st.subheader("🌳 Hierarchical Clustering – Experiment Mode")

    # 🎛️ User controls
    n_samples = st.slider("🔢 Number of Data Points", 10, 150, 50, step=10)
    n_clusters = st.slider("🔗 Number of Clusters", 2, 10, 3)
    linkage_method = st.selectbox("🧬 Linkage Method", ["ward", "complete", "average", "single"])

    # 🧪 Train Model
    X, linked, labels = train_hierarchical_model(n_samples, n_clusters, linkage_method)

    # 📊 Cluster Plot
    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    df["Cluster"] = labels.astype(str)

    fig = px.scatter(df, x="Feature 1", y="Feature 2", color="Cluster",
                     title=f"Hierarchical Clustering ({linkage_method} linkage)",
                     height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 🌲 Dendrogram
    st.subheader("🧬 Dendrogram")
    fig_dendro, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linked, ax=ax)
    st.pyplot(fig_dendro)

    # 📘 Theory
    with st.expander("📘 Hierarchical Clustering – Theory"):
        st.markdown("""
**Hierarchical Clustering** builds a tree (dendrogram) to group similar data points.

- **Agglomerative (bottom-up)**: Start with single points → merge step by step.
- **Divisive (top-down)**: Start with one cluster → split recursively.

### 🔗 Linkage Methods:
- **Ward**: Minimizes total variance (compact clusters)
- **Complete**: Max distance between elements in clusters
- **Average**: Mean distance between points
- **Single**: Minimum distance (prone to chaining)

### ✅ Pros:
- No need to specify number of clusters in advance
- Reveals cluster structure

### ⚠️ Cons:
- Computationally expensive
- Sensitive to noisy data
""")

    # 🔊 Voiceover
    if st.button("🔈 Hear Theory Explanation"):
        st.info("🎙️ Playing voice explanation for Hierarchical Clustering...")
        # Call speak_theory("hierarchical") or similar here
