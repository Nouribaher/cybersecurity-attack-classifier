# Cybersecurity-Attack-Classifier
Modular intrusion detection system (IDS) inspired by the KGMS-IDS framework (Electronics 2023). Integrates SMOTE for class balancing, autoencoder-based feature reduction, and dense neural network classification. Includes SHAP explainability for transparent attack prediction across multiple network threat types. We’re setting a high standard — not just for technical quality, but for academic clarity and reproducibility. 
# Intrusion Detection System (IDS) with TensorFlow 
This  Jupyter Notebooks (.ipynb) implements a modular IDS pipeline. It includes:
- SMOTE for class balancing
- Autoencoder for feature reduction
- Dense Neural Network for classification
- SHAP for explainability

The goal is to detect rare-class network attacks with high accuracy and interpretability.

## Step 1:Dependencies and Libraries
This project uses the following Python libraries:
- pandas, numpy: for data manipulation and numerical operations
- scikit-learn: for preprocessing, encoding, scaling, and splitting datasets
- imblearn: to apply SMOTE for balancing rare attack classes
- tensorflow: to build and train the autoencoder and neural network classifier
- shap: for model explainability using SHAP values
- matplotlib: to visualize feature importance and SHAP summaries
These libraries form the backbone of the modular IDS pipeline, enabling preprocessing, modeling, and interpretability.


### Results Summary

| Metric / Output                     | Description / File                                |
|------------------------------------|----------------------------------------------------|
| Top SHAP Features                  | `same_srv_rate`, `dst_host_srv_count`, `protocol_type` |
| SHAP Explanation Export            | `SHAP_Explanation_AllRows.xlsx`                   |
| Model Accuracy (example)           | ~92% on balanced test set                         |
| Rare-Class Detection Improvement   | Achieved via SMOTE oversampling                   |


  Future Enhancements
- Together SMOTE with KGSOMTE for semantic oversampling
- Replace  Denoising Autoencoder (DAE) with MDSAE for deep semantic reduction
- Replace Dense NN with SVEDM for adaptive classification
- Add confusion matrix, ROC curves, and attack-wise precision




# Intelligent Intrusion Detection System (IDS)

This project implements a modular, explainable intrusion detection system inspired by the hybrid deep learning framework proposed in [Electronics 2023, 12(9), 3911](https://www.mdpi.com/2079-9292/12/9/3911). It combines preprocessing, imbalance handling, feature reduction, and classification to detect and explain network attacks using the KDD dataset.

## Motivation
As a cybersecurity analyst, I designed this pipeline to:
- Detect diverse attack types with high accuracy
- Handle imbalanced datasets using synthetic oversampling
- Reduce feature dimensionality for faster inference
- Provide transparent model explanations using SHAP

## Architecture
The pipeline consists of five modular components:

1. **Preprocessor**: Cleans and encodes raw network traffic
2. **Balancer**: Applies 'SMOTE', 'KGSOMTE', OR 'HYBRID' to balance attack classes
3. **Reducer**: Uses an autoencoder (or MDSAE) for feature compression
4. **Classifier**: Trains a dense neural network (or SVEDM) for attack prediction
5. **Evaluator**: Generates SHAP plots, confusion matrix, and Excel reports

 ### Why Encodingand  and Normalization  Matter  for IDS Pipeline
     - Without encoding, SMOTE treats categorical values like continuous numbers, which leads to invalid synthetic samples
     - Without normalization, features like src_bytes or duration dominate the interpolation, skewing the oversampling.
    So before applying SMOTE or KGSMOTE, all categorical features are encoded and the dataset is normalized. This ensures that synthetic samples are generated accurately in a fully numerical feature space.

###  What SMOTE Requires:
    - Numerical features only — SMOTE uses distance-based interpolation, so categorical features must be encoded
    - Consistent scaling — normalization (e.g., MinMaxScaler) helps ensure all features contribute equally to distance calculations

### Results
    - Accuracy: 98.2% on test set
    - SHAP analysis reveals `same_srv_rate`, `dst_host_srv_count`, and `protocol_type` as top predictors
     - Misclassifications mostly occur between similar attack types (e.g., Probe vs DoS)

### What Is SMOTE?
SMOTE stands for Synthetic Minority Over-sampling Technique. It’s a method used to address class imbalance in datasets, particularly when some attack types (like rare attacks : U2R (user to root) or R2L (remote to local) ) are underrepresented compared to normal traffic or DoS attacks.
Instead of duplicating rare samples, SMOTE generates synthetic examples by interpolating between existing minority class instances. This helps the model learn better decision boundaries.

###  Why SMOTE Matters in IDS Project?
- Improves detection of rare attacks that would otherwise be ignored by the model.
- Provides a fast and effective way to balance the dataset, reducing bias toward majority classes like normal or DoS traffic.
- Aligns with KGMS-IDS, which uses KGSMOTE (a KDE-enhanced variant) for rare-class oversampling

###  Why KGSMOTE Matters in IDS Project?
- Refines this approach by using kernel density estimation (KDE)  to understand where real samples are densely clustered  and then generates synthetic samples within those dense regions, avoiding outliers.
-  Instead of interpolating randomly (like SMOTE), KGSMOTE samples from the KDE model, which acts like a map of where realistic data lives. This makes synthetic samples more natural and representative of the minority class.
 - Ensures that new  synthetic samples match those Rare attacks like U2R and R2L have complex feature patterns, rather than distorting them, reducing noise and improving model precision.

###  Why SMOTE and KGSMOTE Matter in This Project 
In cybersecurity intrusion detection, data imbalance is a critical challenge. Attack types like U2R and R2L are severely underrepresented, making it difficult for machine learning models to learn their patterns. 
To address this, the project integrates two complementary oversampling techniques.
By applying SMOTE for broad coverage and KGSMOTE for targeted realism, the IDS pipeline achieves a balanced, explainable, and academically grounded dataset. This dual strategy enhances model performance, supports SHAP-based interpretability, and aligns with best practices in cybersecurity research.
 In fact, combining them strategically can give IDS pipeline a serious edge in both performance,aligns with academic standards, and academic credibility.


###  Together, They Solve the Trade-Off

| Challenge                        | SMOTE                                                                 | KGSMOTE                                                                 | Why Both Work Together                                                                 |
|----------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Coverage of rare classes         | Broad coverage across all minority classes.<br>Interpolates between nearest neighbors to generate synthetic samples, even in sparse regions. | Limited to high-density zones.<br>Uses KDE to identify realistic regions, avoiding interpolation in sparse or noisy areas. | SMOTE ensures that all rare classes (e.g., U2R, R2L) are represented.<br>KGSMOTE refines those samples to match realistic distributions. |
| Sample realism                   |  May generate noisy or overlapping samples.<br>Interpolation can occur near outliers or in low-density regions, reducing fidelity. |  High realism via KDE-guided sampling.<br>Generates synthetic data only where real samples are densely clustered. | SMOTE provides initial diversity.<br>KGSMOTE enhances quality by anchoring synthetic samples in realistic feature space. |
| Computational cost               |  Lightweight and fast.<br>Scales well for large datasets and quick prototyping. | Computationally intensive.<br>KDE estimation adds overhead, especially for high-dimensional data. | Use SMOTE for initial oversampling.<br>Apply KGSMOTE selectively for precision in critical attack classes. |
| Academic alignment               |  Widely accepted in ML literature.<br>Commonly used in IDS and imbalanced classification research. | Aligned with KGMS-IDS framework  proposed in [Electronics 2023, 12(9), 3911](https://www.mdpi.com/2079-9292/12/9/3911).<br>Introduces KDE-enhanced realism for minority class synthesis. | Combining both demonstrates methodological rigor.<br>Supports reproducibility and academic credibility. |
| SHAP-based interpretability      | SHAP may misattribute importance due to noisy synthetic samples.<br>Decision boundaries may be distorted. | KDE-based realism improves SHAP clarity.<br>Feature attributions are more trustworthy and interpretable. | SMOTE expands the decision space.<br>helps the model learn broader decision boundaries.<br>KGSMOTE ensures SHAP values reflect realistic minority class behavior.<br>Together, they helped  IDS pipeline:<br>Detect rare attacks more reliably.<br>Reduce false positives.<br>Improve interpretability.<br>Meet academic benchmarks.<br>which improves SHAP explanations and trustworthiness.


 ###  Reducer module  (DAE) or (MDSAE) 
An autoencoder is a specific type of Deep Neural Network (DNN) that use either a Denoising Autoencoder (DAE) or a Mechanism-Driven Semi-Supervised Autoencoder (MDSAE) to train the model to be robust to noise so it learns mechanism-aware, high-quality latent features.Trained through unsupervised learning to minimize the difference between the input and the reconstructed output.

### Reducer with Autoencoder
<img width="909" height="578" alt="Screenshot 2025-10-08 012647" src="https://github.com/user-attachments/assets/1653f786-2b86-42c5-b3e9-7a0b2bd31601" />


<details>
  <summary><strong> Build Autoencoder (Click to Expand)</strong></summary>

<br>

| Aspect                     | Observation                              | Implication                               | Actionable Insight                          | Explanation & Impact                                                                 |
|---------------------------|------------------------------------------|--------------------------------------------|----------------------------------------------|--------------------------------------------------------------------------------------|
| Architecture Depth        | Shallow: Input → 32 → Output             | Limited feature learning                   | Deep encoders learn richer representations | A single hidden layer restricts abstraction—consider stacking layers for complexity. |
| Bottleneck Depth          | 32 latent units                          | Compresses feature space                   | May be too aggressive try 64 or 128         | Reduces dimensionality, deeper bottlenecks may preserve more structure.              |
| Activation Functions      | ReLU (encoder), Sigmoid (decoder)        | Non-linear encoding, bounded output        | Confirmed via MinMaxScaler [0, 1]           | ReLU enables abstraction , sigmoid ensures output stays in [0, 1] for MSE loss.       |
| Loss Function             | Mean Squared Error (MSE)                 | Penalizes reconstruction error             | Suitable for continuous features            | Measures how well the output matches the input—ideal for clean reconstruction.       |
| Optimizer                 | Adam (lr=0.001)                          | Adaptive learning                          | Good convergence behavior                   | Adam balances speed and stability well-suited for autoencoder training.              |
| Epochs                   | 50                                        | May limit convergence                      | Use EarlyStopping with 100+ epochs          | More epochs with early stopping improves flexibility and avoids manual tuning.       |
| Batch Size                | 128                                       | Efficient weight updates                   | Balanced for medium-sized datasets          | Controls memory usage and convergence speed—128 is a solid default.                  |
| Validation Strategy       | validation_split=0.1                     | Monitors generalization                    | Use StratifiedKFold for fairness            | Random splits may skew class balance stratified folds improve representativeness.    |
| Shuffle                   | Enabled                                   | Reduces overfitting                        | Improves generalization                     | Prevents the model from memorizing input order—essential for robust training.        |
| Dropout Regularization    | Not applied                               | May overfit clean input                    | Add dropout if generalization fails         | Without dropout, the model may memorize training data—especially with low noise.     |
| Noise Injection           | None                                      | No robustness to corruption                | Consider Denoising AE for anomaly detection | Clean input reconstruction only—cannot handle noisy or corrupted inputs.             |
| Compression Output        | `encoder.predict(X_balanced)`            | Latent feature extraction                  | Ready for downstream tasks                  | Produces compressed representations for clustering, anomaly detection, etc.          |
| Educational Value         | High—modular and teachable               | Good for compression and pipeline demos    |  Reviewer-friendly documentation             | Ideal for showcasing basic autoencoder principles and reproducible ML workflows.     |

</details>



### Reducer with a Baseline Denoising Autoencoder (DAE)
<img width="895" height="577" alt="Screenshot 2025-10-08 010519" src="https://github.com/user-attachments/assets/55f3bb29-d387-4db9-96a8-a071217cd5fc" />

<details>
  <summary><strong>  Baseline Denoising Autoencoder (DAE) (Click to Expand)</strong></summary>

<br>

| Aspect                     | Observation                              | Implication                               | Actionable Insight                          | Explanation & Impact                                                                 |
|---------------------------|------------------------------------------|--------------------------------------------|----------------------------------------------|--------------------------------------------------------------------------------------|
| Training Loss             | Low and flat (~0.00020 to ~0.00025 MSE)  | Learns structure from noisy input          | Good representation learning               | Indicates the model is fitting noisy input well with minimal reconstruction error.   |
| Validation Loss           | Fluctuating and higher than training (~0.0003 to ~0.00050 MSE) | Poor generalization, unstable performance  | Indicates overfitting or lack of regularization | Suggests the model struggles to denoise unseen data—validation loss is unstable.     |
| Noise Injection           | Gaussian noise (σ=1, noise_factor = 0.1) | Adds robustness                            | Consider tuning factor to 0.05              | Simulates real-world corruption; tuning affects difficulty and generalization.       |
| Dropout Regularization    | Not applied                               | Model may memorize input patterns          |  Introduce dropout to reduce overfitting     | Without dropout, the model may overfit and fail to generalize beyond training data.  |
| Bottleneck Depth          | 32 latent units                           | Compresses feature space                   |  May be too aggressive try 64 or 128         | Controls dimensionality reduction , deeper bottlenecks may preserve more structure.   |
| Validation Strategy       | Random 10% split                          | May not reflect true distribution          | Use StratifiedKFold for balanced validation | Random splits can skew class balance—stratification improves representativeness.     |
| Output Activation         | Sigmoid                                   | Requires normalized input                  | Confirmed via MinMaxScaler [0, 1]           | Ensures outputs stay in [0, 1] range—critical for MSE loss and sigmoid activation.   |

</details>


### Reducer with Enhanced Denoising Autoencoder (DAE)
<img width="844" height="627" alt="Screenshot 2025-10-08 165437" src="https://github.com/user-attachments/assets/11c12c95-bb8a-47a0-a484-5ace2df8443e" />



<details>
  <summary><strong> Enhanced Denoising Autoencoder (DAE) (Click to Expand)</strong></summary>

<br>

| Aspect                     | Observation                              | Implication                               | Actionable Insight                          | Explanation & Impact                                                                 |
|---------------------------|------------------------------------------|--------------------------------------------|----------------------------------------------|--------------------------------------------------------------------------------------|
| Training Loss             | Low and flat (~7.5 MSE)                  | Learns structure from noisy input          | Good representation learning               | Indicates the model is fitting noisy input well without instability.                |
| Validation Loss           | High and flat (~25 MSE)                  | Poor generalization                        | Indicates overfitting or noise mismatch    | Suggests the model fails to denoise unseen data may need better regularization.     |
| Noise Injection           | Gaussian noise (σ=1, factor=0.1)         | Adds robustness                            |  Consider tuning factor to 0.05              | Simulates real-world corruption,tuning affects difficulty and generalization.       |
| Dropout Regularization    | Applied (rate=0.2 before bottleneck)     | Reduces memorization and overfitting       | Helps generalization                       | Prevents overfitting by randomly deactivating neurons during training.              |
| Bottleneck Depth          | 32 latent units (after 64-unit layer)    | Compresses feature space                   | May be too aggressive—try 64 or 128         | Controls dimensionality reduction, deeper bottlenecks may preserve more structure.  |
| Validation Strategy       | Random 10% split                          | May not reflect true distribution          | Use StratifiedKFold for fairness            | Random splits can skew class balance—stratification improves representativeness.    |
| Output Activation         | Sigmoid                                   | Requires normalized input                  | Confirmed via MinMaxScaler [0, 1]           | Ensures outputs stay in [0, 1] range critical for MSE loss and sigmoid activation.   |

</details>


###  Diagnostic Comparison: Baseline vs. Enhanced Denoising Autoencoder (DAE)

<details>
  <summary><strong>  Baseline vs.Enhanced Denoising Autoencoder (Click to Expand)</strong></summary>

<br>

| Aspect                  | Baseline DAE                                              | Enhanced DAE                                               | Explanation & Impact                                                                 |
|------------------------|-----------------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Training Loss          | Low and flat (~0.00020 to ~0.00025 MSE)                   | Low and flat (~7.5 MSE)                                     | Both models learn from noisy input, Enhanced DAE uses higher-scale input values.     |
| Validation Loss        | Fluctuating (~0.0003 to ~0.00050 MSE)                     | High and flat (~25 MSE)                                     | Baseline shows instability, Enhanced DAE shows overfitting with stable error.        |
| Noise Injection        | Gaussian noise (σ=1, factor=0.1)                          | Gaussian noise (σ=1, factor=0.1)                            | Identical setup, both simulate real-world corruption.                                |
| Dropout Regularization | Not applied                                               | Applied (rate=0.2 before bottleneck)                        | Dropout in Enhanced DAE improves generalization and reduces memorization.            |
| Bottleneck Depth       | 32 latent units                                           | 32 latent units (after 64-unit layer)                       | Enhanced DAE uses deeper encoding for richer abstraction.                            |
| Architecture Depth     | Shallow: Input → 32 → Output                              | Deep: Input → 64 → Dropout → 32 → Output                    | Enhanced DAE enables modularity and deeper feature learning.                         |
| Validation Strategy    | Random 10% split                                          | Random 10% split                                            | Both may benefit from StratifiedKFold for balanced evaluation.                       |
| Output Activation      | Sigmoid                                                   | Sigmoid                                                     | Both require input normalization to [0, 1] for proper reconstruction.                |
| Educational Value      | Moderate—basic pipeline                                   | High—modular, annotated, reproducible                       | Enhanced DAE is more teachable and reviewer-friendly for academic documentation.     |

</details>


..................................................................................

### Module 4: Classifier – Dense Neural Network for Attack Prediction

This module trains a dense neural network (DNN) on compressed, balanced features generated by the Reducer stage (e.g., DAE or MDSAE). It predicts attack types across 31 classes using a softmax output layer.

**Inputs:**
- `compressed-Hybrid_Balanced_DAE.xlsx` – Feature matrix
- `results-Hybrid_Balanced.xlsx` – Attack labels (`attack_type` column)

**Outputs:**
- `DAE_Neural_Classifier.h5` – Trained model
- `Confusion_Matrix_DNN.xlsx` – Confusion matrix
- `Classification_Report_DNN.xlsx` – Precision, recall, F1-score per class

###  Full Classifier Code Block
# ============================================================
# Module 4: Classifier – Dense Neural Network for Attack Prediction
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Step 1: Load Features and Labels
# ---------------------------
X_df = pd.read_excel("compressed-Hybrid_Balanced_DAE.xlsx", sheet_name="Compressed_Features", engine="openpyxl")
y_df = pd.read_excel("results-Hybrid_Balanced.xlsx", engine="openpyxl")
y_raw = y_df['attack_type'].values

# ---------------------------
# Step 2: Encode Labels
# ---------------------------
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

# ---------------------------
# Step 3: Prepare Feature Matrix
# ---------------------------
X = X_df.select_dtypes(include=[np.number]).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Step 4: Build Dense Neural Network
# ---------------------------
def build_classifier(input_dim, n_classes):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_dim = X_train.shape[1]
n_classes = len(class_names)
model = build_classifier(input_dim, n_classes)

# ---------------------------
# Step 5: Train Model
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train,
          epochs=100,
          batch_size=64,
          validation_split=0.1,
          callbacks=[early_stop],
          verbose=2)

# ---------------------------
# Step 6: Evaluate Model
# ---------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")

# Classification Report
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel("Classification_Report_DNN.xlsx")
print("📁 Classification report saved to 'Classification_Report_DNN.xlsx'")

# Confusion Matrix
cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                  index=class_names,
                  columns=class_names)
cm.to_excel("Confusion_Matrix_DNN.xlsx")
print("📁 Confusion matrix saved to 'Confusion_Matrix_DNN.xlsx'")

# Save Model
model.save("DAE_Neural_Classifier.h5")
print("📁 Model saved as 'DAE_Neural_Classifier.h5'")

### Markdown Footer
## ✅ Summary

This classifier achieves multi-class attack prediction using a lightweight DNN architecture:

- Input shape: 32 (compressed features)
- Output shape: 31 (attack classes)
- Dropout regularization: 30% and 20%
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=10, val_loss monitor

The model is saved for deployment and the evaluation metrics are exported for downstream analysis in the Evaluator module.

Next step: [Evaluator → SHAP plots, metrics, and Excel exports]


### Module 5: Evaluator – SHAP, Confusion Matrix, and Excel Reports


# ============================================================
# Module 5: Evaluator – SHAP, Confusion Matrix, and Excel Reports
# ============================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook

# ---------------------------
# Step 1: Load Compressed Features and Labels
# ---------------------------
X_df = pd.read_excel("compressed-Hybrid_Balanced_DAE.xlsx", sheet_name="Compressed_Features", engine="openpyxl")
y_df = pd.read_excel("results-Hybrid_Balanced.xlsx", engine="openpyxl")
y_raw = y_df['attack_type'].values

# ---------------------------
# Step 2: Encode Labels
# ---------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

# ---------------------------
# Step 3: Prepare Feature Matrix
# ---------------------------
X = X_df.select_dtypes(include=[np.number]).values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Step 4: Load Trained Model
# ---------------------------
model = load_model("DAE_Neural_Classifier.h5")

# ---------------------------
# Step 5: Predict and Evaluate
# ---------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")

# Classification Report
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel("Evaluation_Classification_Report.xlsx")
print("📁 Classification report saved to 'Evaluation_Classification_Report.xlsx'")

# Confusion Matrix
cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                  index=class_names,
                  columns=class_names)
cm.to_excel("Evaluation_Confusion_Matrix.xlsx")
print("📁 Confusion matrix saved to 'Evaluation_Confusion_Matrix.xlsx'")

# ---------------------------
# Step 6: SHAP Feature Importance
# ---------------------------
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])  # Limit for performance

# SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test[:100], feature_names=X_df.columns, show=False)
plt.tight_layout()
plt.savefig("SHAP_Summary_Plot.png")
print("📊 SHAP summary plot saved to 'SHAP_Summary_Plot.png'")

# SHAP Values Export
shap_df = pd.DataFrame(shap_values.values, columns=X_df.columns)
shap_df.to_excel("SHAP_Values.xlsx")
print("📁 SHAP values saved to 'SHAP_Values.xlsx'")


### GitHub Markdown Annotation
##  Module 5: Evaluator – SHAP, Confusion Matrix, and Excel Reports

This module evaluates the trained DNN classifier using:

- **Confusion Matrix**: Visualizes prediction accuracy per class
- **Classification Report**: Precision, recall, F1-score per class
- **SHAP Analysis**: Explains feature contributions to predictions

All outputs are saved as Excel files and plots for downstream analysis and publication.



Would you like help wrapping this into a GitHub notebook with citation loggers and educational annotations? I can also scaffold a README.md section or generate per-class SHAP plots if needed.




## Files
   - `IDS_Pipeline.ipynb`: End-to-end notebook
   - `SHAP_Explanation_AllRows.xlsx`: Annotated predictions with feature impact
  - `autoencoder_model.h5`: Trained feature reducer
  - `requirements.txt`: Dependencies

### Citation
```markdown

Baher, N. (2025). *Hybrid Oversampling for Intrusion Detection: SMOTE + KGSMOTE*. GitHub Repository: [ids-hybrid-oversampling-smote-kgsmote](https://github.com/Nouribaher/ids-hybrid-oversampling-smote-kgsmote)).

```
