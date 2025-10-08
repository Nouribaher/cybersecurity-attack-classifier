import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Store all mappings in a list
mapping_data = []

# Build mappings for each categorical column
for col in ['protocol_type', 'service', 'flag','label']:
    if col in df.columns:
        original_values = df[col].astype(str)
        le = LabelEncoder()
        le.fit(original_values)

        # Build mapping: encoded → original
        for code, name in zip(le.transform(le.classes_), le.classes_):
            mapping_data.append((col, code, name))

# Create a combined DataFrame
mapping_df = pd.DataFrame(mapping_data, columns=['Category', 'Encoded Value', 'Original Name'])

# Display as a single table
print("\n Combined Mapping Table:")
print(mapping_df.to_string(index=False))
# Save to a new Excel file
mapping_df.to_excel('KDDTest_CombinedMapping.xlsx', index=False)
