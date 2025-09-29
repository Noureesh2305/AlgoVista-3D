# experiment_mode/random_forest_experiment.py

import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def render():
    st.subheader("🌳 Random Forest – Experiment Mode")

    st.markdown("""
    **Random Forest** is an ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

    Try experimenting with:
    - Number of Trees (n_estimators)
    - Tree Depth (max_depth)
    - Label Noise (flip_y)
    - Test Split Ratio
    """)

    # Controls
    n_estimators = st.slider("🌲 Number of Trees (n_estimators)", 10, 200, 100, step=10)
    max_depth = st.slider("🌿 Max Depth of Each Tree", 1, 20, 5)
    noise = st.slider("🔀 Label Noise (flip_y)", 0.0, 1.0, 0.05, step=0.01)  # ✅ Valid range [0.0, 1.0]
    test_size = st.slider("🧪 Test Set Size", 0.1, 0.5, 0.3, step=0.05)

    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=300,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=noise,
        random_state=42
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model Accuracy: {acc * 100:.2f}%")

    # Plot 3D data colored by prediction
    fig = px.scatter_3d(
        x=X_test[:, 0], y=X_test[:, 1], z=X_test[:, 2],
        color=y_pred.astype(str),
        title="🎯 3D Random Forest Prediction Visualization",
        labels={"x": "Feature 1", "y": "Feature 2", "z": "Feature 3"},
        opacity=0.8
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("🔍 Adjust the sliders above to see how changes in parameters affect Random Forest performance and predictions.")
