from sklearn.neighbors import KernelDensity

# Choose one minority class to simulate KGSMOTE
minority_class_id = valid_classes[-1]  # pick last valid class
minority_df = df[df['label'] == minority_class_id]
X_minority = scaler.transform(minority_df.drop(columns=['label', 'label_name']))

# Fit KDE and sample synthetic points
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_minority)
synthetic = kde.sample(n_samples=500)

synthetic_df = pd.DataFrame(synthetic, columns=X_raw.columns)
synthetic_df['label'] = minority_class_id
synthetic_df['label_name'] = le.inverse_transform([minority_class_id])[0]

# Combine with original data
df_kgsmote = pd.concat([df, synthetic_df], ignore_index=True)
df_kgsmote.to_excel('KGSMOTE_Balanced.xlsx', index=False)
