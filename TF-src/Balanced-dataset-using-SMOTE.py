from collections import Counter

# Keep only classes with at least 3 samples
counts = Counter(y)
valid_classes = [cls for cls, count in counts.items() if count >= 3]
mask = df['label'].isin(valid_classes)

X_filtered = X_scaled[mask]
y_filtered = y[mask]

from imblearn.over_sampling import SMOTE
# Balanced dataset using SMOTE (safe config)
smote = SMOTE(k_neighbors=2)
X_smote, y_smote = smote.fit_resample(X_filtered, y_filtered)

df_smote = pd.DataFrame(X_smote, columns=X_raw.columns)
df_smote['label'] = y_smote
df_smote['label_name'] = le.inverse_transform(y_smote)

df_smote.to_excel('SMOTE_Balanced.xlsx', index=False)

# Balanced dataset using SMOTE (safe config)
df_smote.label.value_counts()
