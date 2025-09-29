# visuals/plot_apriori.py

import streamlit as st
import plotly.express as px
from models.apriori import get_rules

def render():
    st.header("🛒 Apriori Algorithm – Market Basket Analysis")

    st.markdown("""
    **Apriori** is an algorithm for frequent itemset mining and association rule learning.

    It's useful for:
    - Market basket analysis
    - Recommendation systems
    - Retail and inventory planning

    It works by identifying itemsets that frequently occur together in transactions.
    """)

    st.subheader("🎛️ Experiment Mode: Adjust Parameters")
    min_support = st.slider("📉 Minimum Support", 0.1, 1.0, 0.3, 0.1)
    min_confidence = st.slider("📈 Minimum Confidence", 0.1, 1.0, 0.7, 0.1)

    itemsets, rules = get_rules(min_support, min_confidence)

    st.subheader("✅ Frequent Itemsets")
    st.dataframe(itemsets)

    # Convert frozensets to readable strings
    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

        st.subheader("🔗 Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        fig = px.scatter(rules, x="support", y="confidence", size="lift",
                         hover_data=['antecedents', 'consequents'], color='lift')
        fig.update_layout(title="📊 Association Rules Visualization", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No rules found for the selected support and confidence thresholds.")
  