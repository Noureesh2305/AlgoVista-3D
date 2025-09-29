import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from tutor.voice_guide import speak_theory
import numpy as np

def train_svm_model(C=1.0, kernel='rbf', gamma='scale'):
    X, y = make_classification(n_samples=200, n_features=3, n_informative=3, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return X_train, X_test, y_train, y_pred, model, X, y

def plot_rotating_svm(X, y, angle):
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df['Label'] = y
    fig = px.scatter_3d(df, x='Feature 1', y='Feature 2', z='Feature 3',
                        color=df['Label'].astype(str), title="SVM – 3D Data with Rotating View")
    fig.update_layout(scene_camera=dict(eye=dict(x=np.cos(angle), y=np.sin(angle), z=0.7)))
    return fig

def render():
    st.subheader("📌 Support Vector Machine (SVM) – Interactive Visualizer")

    # Sidebar experiment mode
    st.sidebar.header("🔬 Experiment Settings")
    C = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0, step=0.1)
    kernel = st.sidebar.selectbox("Kernel Type", ['linear', 'rbf', 'poly', 'sigmoid'])
    gamma = st.sidebar.selectbox("Gamma", ['scale', 'auto'])

    # Train and visualize
    X_train, X_test, y_train, y_pred, model, X, y = train_svm_model(C, kernel, gamma)

    # Training plot
    st.markdown("### 🎯 Training Data")
    df_train = pd.DataFrame(X_train, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_train['Label'] = y_train
    fig_train = px.scatter_3d(df_train, x='Feature 1', y='Feature 2', z='Feature 3',
                              color=df_train['Label'].astype(str),
                              title="SVM - Training Data")
    st.plotly_chart(fig_train, use_container_width=True)

    # Test predictions
    st.markdown("### 🤖 Predictions on Test Data")
    df_test = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_test['Predicted'] = y_pred
    fig_test = px.scatter_3d(df_test, x='Feature 1', y='Feature 2', z='Feature 3',
                             color=df_test['Predicted'].astype(str),
                             title="SVM - Test Predictions")
    st.plotly_chart(fig_test, use_container_width=True)

    # Optional 3D animation (rotate view)
    st.markdown("### 🎥 Rotating SVM Visualization")
    rotation_angle = st.slider("Rotation Angle (0 → 2π)", 0.0, 6.28, 1.57, step=0.1)
    fig_rotate = plot_rotating_svm(X, y, angle=rotation_angle)
    st.plotly_chart(fig_rotate, use_container_width=True)

    # Theory section
    st.markdown("### 📘 SVM – Theory")
    st.markdown("""
**Support Vector Machine (SVM)** is a powerful **supervised learning** algorithm used for **classification** and **regression** tasks.

#### 🧠 Key Concepts:
- Finds the **optimal separating hyperplane** with **maximum margin** between classes.
- Relies on **support vectors** – critical boundary data points.
- Can use **kernel tricks** to solve non-linear problems.

#### 🧪 Kernels:
- **Linear**: Straight-line decision boundaries
- **RBF / Polynomial / Sigmoid**: Curved boundaries for complex data

#### ✅ Advantages:
- Effective in **high-dimensional spaces**
- Good for **clear margin separation**

#### ⚠️ Limitations:
- Slow for large datasets
- Less effective on overlapping classes
    """)

    # Voiceover
    if st.button("🔊 Explain with Voice"):
        speak_theory("Support Vector Machine (SVM)")
