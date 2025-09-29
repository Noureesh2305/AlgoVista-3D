import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tutor.voice_guide import speak_theory

# 📊 Train Decision Tree with customizable parameters
def train_decision_tree_model(max_depth, criterion, min_samples_split):
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                   min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return X_train, X_test, y_train, y_pred, model, X, y

# 🌐 Plot 3D decision surface
def plot_3d_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # 3D surface boundary
    fig.add_trace(go.Surface(
        z=Z, x=xx, y=yy, colorscale='Viridis',
        showscale=False, opacity=0.6,
        name="Decision Surface"
    ))

    # Actual data points
    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y,
        mode='markers',
        marker=dict(size=4, color=y, colorscale='Viridis'),
        name="Data Points"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Class',
        ),
        title="🌐 3D Decision Tree Visualization",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

# 📘 Main Render Function
def render():
    st.header("🌳 Decision Tree Classifier – Learn & Experiment")

    # 🧪 Experiment Mode – Sidebar Controls
    st.sidebar.header("🧪 Experiment Mode Settings")
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 4)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)

    # 🎓 Voiceover Button
    if st.button("🔊 Voiceover: Explain Decision Tree"):
        speak_theory("Decision Tree")

    # 📘 Theory
    st.markdown("### 📘 Decision Tree – Theory")
    st.markdown("""
**Decision Tree** is a supervised machine learning algorithm used for classification and regression tasks.  
It splits the dataset into subsets based on the value of input features. This is done recursively, forming a tree-like model of decisions.

#### 🧠 Key Concepts:
- **Root Node**: First feature split  
- **Decision Node**: Internal node that splits further  
- **Leaf Node**: Final class label  
- **Gini/Entropy**: Used to measure split quality

#### ✅ Advantages:
- Easy to understand and visualize  
- Handles both numerical and categorical data

#### ⚠️ Disadvantages:
- Prone to overfitting  
- Sensitive to small data changes  
""")

    # 🔁 Train model with user settings
    X_train, X_test, y_train, y_pred, model, X, y = train_decision_tree_model(
        max_depth, criterion, min_samples_split
    )

    # 🌐 3D Visualization
    st.subheader("🌐 3D Decision Boundary")
    fig = plot_3d_decision_boundary(model, X, y)
    st.plotly_chart(fig)

    # 📊 2D Tree Diagram
    st.subheader("📊 2D Tree Structure")
    fig2, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model, filled=True, feature_names=["Feature 1", "Feature 2"], class_names=["Class 0", "Class 1"])
    st.pyplot(fig2)
