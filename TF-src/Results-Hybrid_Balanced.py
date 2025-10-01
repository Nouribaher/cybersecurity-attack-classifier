X_hybrid, y_hybrid = balance(X_scaled, y, method='hybrid', minority_class_id=5)
df_hybrid = pd.DataFrame(X_hybrid, columns=X_raw.columns)
df_hybrid['label'] = y_hybrid
df_hybrid['label_name'] = le.inverse_transform(y_hybrid)
df_hybrid.to_excel('results-Hybrid_Balanced.xlsx', index=False)
