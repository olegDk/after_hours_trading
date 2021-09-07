import pandas as pd
import numpy as np

res = pd.read_csv('/home/oleh/Downloads/res.csv')

res.head()

md = pd.read_csv('/home/oleh/Downloads/md.csv')

md.head()

res_filtered = res[['Symbol', 'Side', 'Size', 'Price']]

md_filtered = md[['Symbol', 'Open']]

res_md_joined = pd.merge(left=res_filtered,
                         right=md_filtered,
                         on='Symbol',
                         suffixes=('res', 'md'))
res_md_joined['Tier_delta'] = res_md_joined['Open'] - res_md_joined['Price']
res_md_joined['Profit'] = 0
for i, row in res_md_joined.iterrows():
    side = row['Side']
    delta = row['Tier_delta']
    size = row['Size']
    if side == 'T':
        if delta < 0:
            res_md_joined.loc[i, 'Profit'] = size * np.abs(delta)
        else:
            res_md_joined.loc[i, 'Profit'] = size * -delta
    else:
        res_md_joined.loc[i, 'Profit'] = size * delta

res_md_total = res_md_joined.groupby(by='Symbol').agg({
    'Size': 'sum',
    'Profit': 'sum'
}).sort_values(by='Profit', ascending=False)

res_md_total.sum()
