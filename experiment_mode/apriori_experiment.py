import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

def run_apriori_experiment(df, min_support, min_confidence):
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

def render():
    st.title("🛒 Apriori Algorithm (Experiment Mode)")
    st.markdown("""
    The **Apriori algorithm** is used for **market basket analysis** to find frequent itemsets and association rules.
    
    - **Input:** Transaction dataset (one-hot encoded)
    - **Output:** Frequent itemsets and confidence-based rules
    """)

    st.markdown("#### 📥 Sample Dataset: Groceries (One-Hot Encoded)")
    dataset = [
        ['milk', 'bread', 'butter'],
        ['beer', 'bread'],
        ['milk', 'bread', 'butter'],
        ['beer', 'bread'],
        ['milk', 'bread'],
        ['milk', 'bread', 'butter'],
        ['milk', 'bread'],
        ['beer', 'bread']
    ]

    df = pd.DataFrame(dataset)
    one_hot = pd.get_dummies(df.stack()).groupby(level=0).sum()

    st.dataframe(one_hot)

    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.3, 0.1)
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.7, 0.1)

    if st.button("🔍 Run Apriori"):
        freq_items, rules = run_apriori_experiment(one_hot, min_support, min_confidence)

        st.subheader("📊 Frequent Itemsets")
        st.dataframe(freq_items)

        if not freq_items.empty:
            fig = px.bar(freq_items, x='itemsets', y='support', title='Frequent Itemsets')
            st.plotly_chart(fig)

        st.subheader("📐 Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        if not rules.empty:
            fig2 = px.scatter(
                rules,
                x="support",
                y="confidence",
                size="lift",
                hover_data=["antecedents", "consequents"],
                title="Association Rules (Support vs Confidence)"
            )
            st.plotly_chart(fig2)

    st.info("Use sliders to experiment with support & confidence thresholds.")
