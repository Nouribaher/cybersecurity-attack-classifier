#Feature Normalization
# MinMaxScaler transforms numerical features to a fixed range usually [0, 1] using the formula:
# "\text{scaled} = \frac{\text{value} - \text{min}}{\text{max} - \text{min}}"
#This keeps the relative spacing between values but ensures all features are on the same scale, which is crucial for models and layers like Dense, LSTM, SVM, or KNN.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the encoded Excel file
df = pd.read_excel('KDDTest_Encoded.xlsx')

# Separate features and target
#X_raw = df.drop('label', axis=1)
#y = df['label']
X_raw = df.copy()   
# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Convert back to DataFrame
normalized_df = pd.DataFrame(X_scaled, columns=X_raw.columns)
#normalized_df['label'] = y  # Add label back

# Show sample output
print("\n✅ Normalized Data Sample:")
print(normalized_df.head())

# Save to a new Excel file
normalized_df.to_excel('KDDTest_Normalized.xlsx', index=False)
print("\n✅ Normalized data saved to 'KDDTest_Normalized.xlsx'")
