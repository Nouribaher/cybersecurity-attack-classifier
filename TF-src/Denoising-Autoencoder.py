import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# Load the dataset
file_path = "results-Hybrid_Balanced.xlsx"  # Your attached file
df = pd.read_excel(file_path, engine="openpyxl")

# Ensure all features are numeric
X = df.select_dtypes(include=[np.number]).values

# Optional normalization (recommended for autoencoders)
X = (X - X.min()) / (X.max() - X.min())

# ..........................................................................................
# Add Noise (for Denoising Autoencoder)
noise_factor = 0.1
X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
X_noisy = np.clip(X_noisy, 0., 1.)  # Keep in range [0,1]
# ..........................................................................................

#  Build Denoising Autoencoder
def build_denoising_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
 
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder

input_dim = X.shape[1]
autoencoder, encoder = build_denoising_autoencoder(input_dim, encoding_dim=32)


# Train Denoising Autoencoder
autoencoder.fit(X_noisy, X,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_split=0.1,
                verbose=1)

# Compress Features (encoded latent space)
X_compressed = encoder.predict(X)
compressed_df = pd.DataFrame(X_compressed, columns=[f"latent_{i+1}" for i in range(X_compressed.shape[1])])

# Save compressed features and metadata
metadata = pd.DataFrame({
    "original_dim": [input_dim],
    "compressed_dim": [X_compressed.shape[1]],
    "samples": [X.shape[0]],
    "source": [file_path],
    "method": ["Denoising Autoencoder (noise_factor=0.1)"]
})

with pd.ExcelWriter("compressed-Hybrid_Balanced_DAE.xlsx", engine="openpyxl") as writer:
    compressed_df.to_excel(writer, sheet_name="Compressed_Features", index=False)
    metadata.to_excel(writer, sheet_name="Compression_Metadata", index=False)

print("Denoising Autoencoder completed! Results saved to 'compressed-Hybrid_Balanced_DAE.xlsx'")
