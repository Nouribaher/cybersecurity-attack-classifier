# Cleans and encodes the dataset
# We encode attack labels for model compatibility while preserving human-readable names.
# Encoding Categorical Features
- Purpose: Converts string-based categorical features into numeric form.
- Why: ML models (especially neural nets) require numerical input.
- Example: 'tcp', 'udp', 'icmp' → 0, 1, 2 (depending on frequency or order)
# Data shape: (11850, 43)
- Unique encoded values in 'protocol_type': [np.int64(0), np.int64(1), np.int64(2)]
- Unique encoded values in 'service': [np.int64(0), np.int64(1),..... np.int64(61)]
- Unique encoded values in 'flag': [np.int64(0), np.int64(1),........ np.int64(10)]
- Unique encoded values in 'label': [np.int64(0), np.int64(1),....... np.int64(37)]
#  Encoded data saved to 'KDDTest_Encoded.xlsx'


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the Excel file
df = pd.read_excel('KDDTest_Full.xlsx')

# Encode categorical columns (keep original column names)
for col in ['protocol_type', 'service', 'flag','label']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Show the first few rows of the updated DataFrame
print("\n✅ Encoded Data Sample:")
print(df.head())

# Show the shape and column names
print("\n✅ Data shape:", df.shape)
print("✅ Columns:", df.columns.tolist())

# Show unique values in each encoded column
for col in ['protocol_type', 'service', 'flag','label']:
    print(f"\n✅ Unique encoded values in '{col}':", sorted(df[col].unique()))

# Save to a new Excel file
df.to_excel('KDDTest_Encoded.xlsx', index=False)
print("\n✅ Encoded data saved to 'KDDTest_Encoded.xlsx'")
