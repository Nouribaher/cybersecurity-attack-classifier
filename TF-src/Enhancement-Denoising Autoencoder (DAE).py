import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the balanced dataset
df = pd.read_excel("results-Hybrid_Balanced.xlsx", engine="openpyxl")
X_balanced = df.values  # Assumes all columns are features

#  Inject Gaussian Noise for Denoising Autoencoder
noise_factor = 0.1
X_noisy = X_balanced + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_balanced.shape)
X_noisy = np.clip(X_noisy, 0., 1.)  # Keep values in [0, 1]

#  Build Deeper Autoencoder with Dropout
def build_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    bottleneck = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(bottleneck)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=bottleneck)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder

# Train Autoencoder with EarlyStopping
input_dim = X_balanced.shape[1]
autoencoder, encoder = build_autoencoder(input_dim, encoding_dim=32)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = autoencoder.fit(X_noisy, X_balanced,
                          epochs=100,
                          batch_size=128,
                          shuffle=True,
                          validation_split=0.1,
                          verbose=1,
                          callbacks=[early_stop])

#  Compress Features
X_compressed = encoder.predict(X_balanced)
compressed_df = pd.DataFrame(X_compressed, columns=[f"latent_{i+1}" for i in range(X_compressed.shape[1])])

#  Log Reconstruction Error
X_reconstructed = autoencoder.predict(X_balanced)
recon_error = np.mean(np.square(X_balanced - X_reconstructed), axis=1)
error_df = pd.DataFrame({"reconstruction_error": recon_error})

# Compression Metadata
metadata = pd.DataFrame({
    "original_dim": [input_dim],
    "compressed_dim": [X_compressed.shape[1]],
    "samples": [X_balanced.shape[0]],
    "source": ["results-Hybrid_Balanced.xlsx"],
    "method": ["Denoising Autoencoder (SMOTE + KGSMOTE balanced)"]
})

#  Save to Excel with three sheets
with pd.ExcelWriter("compressed-Hybrid_Balanced-Denoising.xlsx", engine="openpyxl") as writer:
    compressed_df.to_excel(writer, sheet_name="Compressed_Features", index=False)
    error_df.to_excel(writer, sheet_name="Reconstruction_Error", index=False)
    metadata.to_excel(writer, sheet_name="Compression_Metadata", index=False)

print(" Denoising autoencoder complete. Saved to 'compressed-Hybrid_Balanced.xlsx'")
