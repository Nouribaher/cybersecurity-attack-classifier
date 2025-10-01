X_kg, y_kg = balance(X_scaled, y, method='kgsmote', minority_class_id=5)
df_kgsmote = pd.DataFrame(X_kg, columns=X_raw.columns)
df_kgsmote['label'] = y_kg
df_kgsmote['label_name'] = le.inverse_transform(y_kg)
df_kgsmote.to_excel('results-KGSMOTE_Balanced.xlsx', index=False)
