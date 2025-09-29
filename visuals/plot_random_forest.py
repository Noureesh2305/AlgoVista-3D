# visuals/plot_random_forest.py

import streamlit as st
import plotly.express as px
import pandas as pd
from models.random_forest import train_random_forest_model

def render():
    st.subheader("📌 Random Forest – Visualization & Theory")

    # Train model and get data
    X_train, X_test, y_train, y_pred, model = train_random_forest_model()

    # Dynamically detect number of features
    num_features = X_train.shape[1]
    feature_cols = [f"Feature {i+1}" for i in range(num_features)]

    # Create dataframes
    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_train['Label'] = y_train

    df_test = pd.DataFrame(X_test, columns=feature_cols)
    df_test['Predicted'] = y_pred

    # Dropdowns to select 3 features for 3D plotting
    st.markdown("### 🧪 Select 3 Features to Visualize")
    feature_x = st.selectbox("X-axis", feature_cols, index=0)
    feature_y = st.selectbox("Y-axis", feature_cols, index=1)
    feature_z = st.selectbox("Z-axis", feature_cols, index=2)

    st.markdown("### 🎯 Training Data")
    fig_train = px.scatter_3d(df_train, x=feature_x, y=feature_y, z=feature_z,
                              color=df_train['Label'].astype(str),
                              title="Random Forest - Training Data")
    st.plotly_chart(fig_train, use_container_width=True)

    st.markdown("### 🤖 Predictions on Test Data")
    fig_test = px.scatter_3d(df_test, x=feature_x, y=feature_y, z=feature_z,
                             color=df_test['Predicted'].astype(str),
                             title="Random Forest - Predictions")
    st.plotly_chart(fig_test, use_container_width=True)

    # 📘 Theory
    st.markdown("### 📘 Random Forest – Theory")
    st.markdown("""
**Random Forest** is an **ensemble learning** method that builds multiple decision trees and merges their predictions.

#### 📎 Key Concepts:
- Trains on random subsets of data and features
- Combines outputs using **majority voting** (classification) or **averaging** (regression)
- Each tree is trained independently → **parallelizable**

#### 🧠 Formula:
- Final prediction = majority(classifiers) or mean(regressors)

#### ✅ Advantages:
- **Reduces overfitting** from individual decision trees
- Works well with large datasets and high dimensions
- Robust and accurate

#### ⚠️ Limitations:
- Less interpretable than a single decision tree
- Slower training with many trees
- Requires tuning parameters like `n_estimators`, `max_depth`
""")
