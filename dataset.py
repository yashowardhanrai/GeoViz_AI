import pandas as pd
df = pd.read_csv(r'data\processed\geoviz_risk_dataset.csv')
print(df.shape)
print(df['risk_score'].describe())
print(df['risk_score'].autocorr(lag=1))