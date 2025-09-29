# experiment_mode/decision_tree_experiment.py

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import numpy as np
from tutor.voice_guide import speak_theory

def render():
    st.markdown("## 🌳 Decision Tree Experiment Mode")
    st.markdown("""
    A Decision Tree splits data recursively based on feature values to predict the target.
    It's easy to visualize and interpret.

    - Try adjusting the tree depth and criterion to observe how it affects accuracy.
    - You can also click 'Grow Tree' to visualize a 3D-style tree structure.
    """)

    # Voiceover button
    if st.button("🔊 Play Voice Explanation"):
        speak_theory("Decision Tree")

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    class_names = data.target_names

    # Parameters
    max_depth = st.slider("🧩 Max Depth of Tree", 1, 10, value=3)
    criterion = st.selectbox("📐 Criterion", ["gini", "entropy"])

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Accuracy: {acc:.2f}")

    # 3D-like Tree Plot using Plotly
    if st.button("🌱 Grow Tree (Visualize Structure)"):
        fig = go.Figure()

        def add_node(fig, text, x, y, parent_x=None, parent_y=None):
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[0],
                text=[text],
                mode="markers+text",
                marker=dict(size=10, color="green"),
                textposition="bottom center"
            ))
            if parent_x is not None:
                fig.add_trace(go.Scatter3d(
                    x=[parent_x, x],
                    y=[parent_y, y],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="lightgray", width=2)
                ))

        def build_3d_tree(clf, feature_names, node_id=0, x=0, y=0, dx=6, depth=0, max_depth=5, parent=None):
            if depth > max_depth or node_id == -1:
                return
            tree_ = clf.tree_
            if tree_.feature[node_id] != -2:  # not a leaf
                name = feature_names[tree_.feature[node_id]]
                threshold = tree_.threshold[node_id]
                label = f"{name} ≤ {threshold:.2f}"
            else:
                label = f"Leaf: {np.argmax(tree_.value[node_id])}"

            if parent:
                add_node(fig, label, x, y, *parent)
            else:
                add_node(fig, label, x, y)

            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]

            build_3d_tree(clf, feature_names, left_child, x - dx, y - 5, dx/1.8, depth + 1, max_depth, (x, y))
            build_3d_tree(clf, feature_names, right_child, x + dx, y - 5, dx/1.8, depth + 1, max_depth, (x, y))

        build_3d_tree(clf, feature_names)

        fig.update_layout(
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
                zaxis=dict(showticklabels=False),
            ),
            title="🌳 Interactive 3D Decision Tree",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("Adjust parameters above to explore how Decision Trees learn.")
