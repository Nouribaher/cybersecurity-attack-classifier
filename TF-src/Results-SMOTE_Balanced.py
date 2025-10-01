X_bal, y_bal = balance(X_scaled, y, method='smote')
df_smote = pd.DataFrame(X_bal, columns=X_raw.columns)
df_smote['label'] = y_bal
df_smote['label_name'] = le.inverse_transform(y_bal)
df_smote.to_excel('results-SMOTE_Balanced.xlsx', index=False)
# Balanced dataset using SMOTE (safe config)
df_smote.label.value_counts()
