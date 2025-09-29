# models/apriori.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def get_rules(min_support=0.3, min_confidence=0.7):
    dataset = [
        ['milk', 'bread', 'butter'],
        ['bread', 'butter'],
        ['milk', 'bread'],
        ['milk', 'butter'],
        ['bread', 'butter'],
        ['milk', 'bread', 'butter'],
    ]

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules
