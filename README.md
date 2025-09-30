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


### 📊 Results Summary

| Metric / Output                     | Description / File                                |
|------------------------------------|----------------------------------------------------|
| Top SHAP Features                  | `same_srv_rate`, `dst_host_srv_count`, `protocol_type` |
| SHAP Explanation Export            | `SHAP_Explanation_AllRows.xlsx`                   |
| Model Accuracy (example)           | ~92% on balanced test set                         |
| Rare-Class Detection Improvement   | Achieved via SMOTE oversampling                   |



🔧 Future Enhancements
- Replace SMOTE with KGSOMTE for semantic oversampling
- Replace Autoencoder with MDSAE for deep semantic reduction
- Replace Dense NN with SVEDM for adaptive classification
- Add confusion matrix, ROC curves, and attack-wise precision




# Intelligent Intrusion Detection System (IDS)

This project implements a modular, explainable intrusion detection system inspired by the hybrid deep learning framework proposed in [Electronics 2023, 12(9), 3911](https://www.mdpi.com/2079-9292/12/9/3911). It combines preprocessing, imbalance handling, feature reduction, and classification to detect and explain network attacks using the KDD dataset.

## 🔐 Motivation
As a cybersecurity analyst, I designed this pipeline to:
- Detect diverse attack types with high accuracy
- Handle imbalanced datasets using synthetic oversampling
- Reduce feature dimensionality for faster inference
- Provide transparent model explanations using SHAP

## 🧱 Architecture
The pipeline consists of five modular components:

1. **Preprocessor**: Cleans and encodes raw network traffic
2. **Balancer**: Applies SMOTE (or KGSOMTE) to balance attack classes
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

## 📁 Files
   - `IDS_Pipeline.ipynb`: End-to-end notebook
   - `SHAP_Explanation_AllRows.xlsx`: Annotated predictions with feature impact
  - `autoencoder_model.h5`: Trained feature reducer
  - `requirements.txt`: Dependencies

### Citation
```markdown

Baher, N. (2025). *Hybrid Oversampling for Intrusion Detection: SMOTE + KGSMOTE*. GitHub Repository: [ids-hybrid-oversampling-smote-kgsmote](https://github.com/Nouribaher/ids-hybrid-oversampling-smote-kgsmote)).

```
