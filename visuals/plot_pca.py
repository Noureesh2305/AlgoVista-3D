# visuals/plot_pca.py

import streamlit as st
import pandas as pd
import plotly.express as px
from models.pca import train_pca_model

def render():
    st.header("📉 Principal Component Analysis (PCA) – Dimensionality Reduction")

    st.markdown("""
**PCA (Principal Component Analysis)** is an **unsupervised** technique to reduce the number of dimensions while retaining most of the variance (information).

Use the slider below to experiment with different numbers of components:
""")

    n_components = st.slider("🔢 Number of Principal Components", min_value=2, max_value=4, value=3, step=1)
    X_train, X_test, y_train, y_test = train_pca_model(n_components)

    df_train = pd.DataFrame(X_train, columns=[f"PC{i+1}" for i in range(n_components)])
    df_train["Label"] = y_train

    df_test = pd.DataFrame(X_test, columns=[f"PC{i+1}" for i in range(n_components)])
    df_test["Label"] = y_test

    if n_components == 3:
        st.markdown("### 🎯 Training Data – 3D Projection")
        fig_train = px.scatter_3d(df_train, x="PC1", y="PC2", z="PC3",
                                  color=df_train["Label"].astype(str),
                                  title="PCA – Training Data (3D)")
        st.plotly_chart(fig_train, use_container_width=True)

        st.markdown("### 🤖 Test Data – 3D Projection")
        fig_test = px.scatter_3d(df_test, x="PC1", y="PC2", z="PC3",
                                 color=df_test["Label"].astype(str),
                                 title="PCA – Test Data (3D)")
        st.plotly_chart(fig_test, use_container_width=True)
    else:
        st.markdown("### 🎯 Training Data – 2D Projection")
        fig_train = px.scatter(df_train, x="PC1", y="PC2",
                               color=df_train["Label"].astype(str),
                               title="PCA – Training Data (2D)")
        st.plotly_chart(fig_train, use_container_width=True)

        st.markdown("### 🤖 Test Data – 2D Projection")
        fig_test = px.scatter(df_test, x="PC1", y="PC2",
                              color=df_test["Label"].astype(str),
                              title="PCA – Test Data (2D)")
        st.plotly_chart(fig_test, use_container_width=True)

    # Theory
    with st.expander("📘 PCA Theory Explanation"):
        st.markdown("""
### 🧠 What is PCA?

PCA stands for **Principal Component Analysis**, a technique to:
- Reduce high-dimensional data to fewer dimensions
- Preserve as much variance (information) as possible
- Remove correlated and redundant features

---

### 🔬 How it Works:
1. Standardize the dataset
2. Compute the covariance matrix
3. Calculate eigenvectors & eigenvalues
4. Project data onto the top-k eigenvectors (principal components)

---

### 📊 Use Cases:
- Visualization of datasets (like Iris)
- Speeding up ML training
- Denoising
- Feature compression

---

📌 PCA is **unsupervised** – it doesn't use the output labels during transformation.
""")
