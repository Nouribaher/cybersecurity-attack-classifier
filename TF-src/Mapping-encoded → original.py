# Show unique values and their original names
for col in [ 'label']:
    if col in df.columns:
        # Create a temporary column with original string values
        original_values = df[col].astype(str)  # Ensure it's string

        # Fit LabelEncoder on original values
        le = LabelEncoder()
        le.fit(original_values)

        # Build mapping: encoded → original
        mapping = dict(zip(le.transform(le.classes_), le.classes_))

        print(f"\n Unique values in '{col}':")
        for code, name in sorted(mapping.items()):
            print(f"  {code:2} → {name}")
