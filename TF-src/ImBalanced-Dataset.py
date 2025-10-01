import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load dataset
df = pd.read_excel('KDDTest_Normalized.xlsx')
# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
df['label_name'] = le.inverse_transform(df['label'])

# Normalize features
X_raw = df.drop(columns=['label', 'label_name'])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)
y = df['label']

#ImBalanced dataset
df.label.value_counts()
