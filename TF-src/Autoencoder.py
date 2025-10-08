import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

#  Load the balanced dataset
df = pd.read_excel("results-Hybrid_Balanced.xlsx", engine="openpyxl")
X_balanced = df.values  # Assumes all columns are features

#  Build Autoencoder
def build_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))   # Rectified Linear Unit (ReLU)is defined by the formula f(x)= max (0,x) and introduces non-linearity
    encoded = Dense(encoding_dim, activation='relu')(input_layer) # compresses input features into a lower-dimensional representation.
    decoded = Dense(input_dim, activation='sigmoid')(encoded)     # reconstructs the original input from the compressed representation.
                                              # σ(x) = 1/(1+exp(-x))  the output is always between 0 and 1
 
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse') # Mean Squared Error (MSE) between original and reconstructed input
    return autoencoder, encoder

    # This setup helps the model learn to reconstruct clean feature vectors from corrupted ones—useful for anomaly detection.
   
# Train Autoencoder
input_dim = X_balanced.shape[1]
autoencoder, encoder = build_autoencoder(input_dim, encoding_dim=32)

autoencoder.fit(X_balanced, X_balanced,
                epochs=50, # It gives the model enough time to converge on clean reconstruction.
                           # Recommendation: Use epochs=100 + EarlyStopping rather than lowering to 50. 
                           # That gives flexibility and ensures you get the best model without manual tuning or overtraining.
                
                batch_size=128, # number of samples the model processes before updating its weights.
                                # If you have 10,000 samples then 1 epoch = about 78 updates (10,000 / 128 ≈ 78 batches).
                
                shuffle=True,    # Randomly shuffle training data before each epoch to reduce overfitting and improve generalization
                
                validation_split=0.1, # Reserves 10% of the training data for validation. Helps monitor performance on unseen data during training.
                
                verbose=1)      # Controls output during training: 0 = silent, 1 = progress bar, 2 = one line per epoch.

#  Compress Features
X_compressed = encoder.predict(X_balanced)
compressed_df = pd.DataFrame(X_compressed, columns=[f"latent_{i+1}" for i in range(X_compressed.shape[1])])

# Compression Metadata
metadata = pd.DataFrame({
    "original_dim": [input_dim],
    "compressed_dim": [X_compressed.shape[1]],
    "samples": [X_balanced.shape[0]],
    "source": ["results-Hybrid_Balanced.xlsx"],
    "method": ["Autoencoder (SMOTE + KGSMOTE balanced)"]
})


# Save to Excel with two sheets
with pd.ExcelWriter("compressed-Hybrid_Balanced.xlsx", engine="openpyxl") as writer:
    compressed_df.to_excel(writer, sheet_name="Compressed_Features", index=False)
    metadata.to_excel(writer, sheet_name="Compression_Metadata", index=False)

print("All-in-one compression complete. Saved to 'compressed-Hybrid_Balanced.xlsx'")
