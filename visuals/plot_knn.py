# visuals/plot_knn.py

import streamlit as st
import plotly.express as px
import pandas as pd
from models.knn import train_knn_model

def render():
    st.subheader("📌 K-Nearest Neighbors (KNN) – Visualization & Theory")
    
    # Train model and get data
    X_train, X_test, y_train, y_pred, model = train_knn_model()

    # Create DataFrame for 3D plotting
    df_train = pd.DataFrame(X_train, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_train['Label'] = y_train

    df_test = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_test['Predicted'] = y_pred

    # 📊 3D Plot for Training Data
    st.markdown("### 🎯 Training Data Visualization")
    fig_train = px.scatter_3d(df_train, x='Feature 1', y='Feature 2', z='Feature 3',
                              color=df_train['Label'].astype(str),
                              title="KNN - Training Data", opacity=0.7)
    st.plotly_chart(fig_train, use_container_width=True)

    # 📊 3D Plot for Test Data with Predictions
    st.markdown("### 🤖 Predictions on Test Data")
    fig_test = px.scatter_3d(df_test, x='Feature 1', y='Feature 2', z='Feature 3',
                             color=df_test['Predicted'].astype(str),
                             title="KNN - Test Data Predictions", symbol=df_test['Predicted'].astype(str))
    st.plotly_chart(fig_test, use_container_width=True)

    # 📘 Theory Explanation
    st.markdown("### 📘 KNN – Theory")
    st.markdown("""
**K-Nearest Neighbors (KNN)** is a **supervised machine learning** algorithm used for both classification and regression. It is a **lazy learner** that stores the training data and makes predictions during inference time.

#### 📎 Key Points:
- **Instance-based** algorithm: No explicit training phase.
- **Distance-based**: Most commonly uses Euclidean distance.
- **K** is the number of neighbors used to classify a point.
- **Majority vote**: In classification, the class most common among neighbors is assigned.
- Sensitive to **feature scaling** and **outliers**.

#### ✅ Advantages:
- Simple and intuitive
- No training time
- Works well with small datasets

#### ⚠️ Limitations:
- Slow with large datasets
- Struggles with high-dimensional data
- Sensitive to irrelevant features
""")
