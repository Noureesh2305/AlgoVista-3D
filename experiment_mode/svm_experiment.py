# experiment_mode/svm_experiment.py

import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import plotly.graph_objects as go

def render():
    st.subheader("⚙️ SVM – Experiment Mode")

    st.markdown("""
    **Support Vector Machines (SVM)** is a supervised learning algorithm that finds an optimal separating hyperplane between classes.
    
    Experiment with:
    - Kernel type
    - Regularization (C)
    - Gamma (for RBF/poly kernels)
    """)

    # Parameter selection
    kernel = st.selectbox("Select Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01)
    gamma = st.slider("Gamma (for RBF/poly/sigmoid)", 0.001, 1.0, 0.1, step=0.001)

    # Load dataset
    X, y = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0,
                                        n_clusters_per_class=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train SVM model
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    # Meshgrid for decision surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        z=Z,
        showscale=False,
        colorscale='RdBu',
        opacity=0.4,
        name='Decision Surface'
    ))

    fig.add_trace(go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode='markers',
        marker=dict(color=y_test, colorscale='Viridis', size=8, line=dict(width=1, color='black')),
        name='Test Data'
    ))

    st.plotly_chart(fig)
    st.markdown("✅ *Change kernel, C, and gamma to observe how the decision boundary changes!*")
