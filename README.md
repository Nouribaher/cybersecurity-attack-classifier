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


### 🔗 Together, They Solve the Trade-Off

| Challenge                        | SMOTE                                                                 | KGSMOTE                                                                 | Why Both Work Together                                                                 |
|----------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Coverage of rare classes         | ✅ Broad coverage across all minority classes.<br>Interpolates between nearest neighbors to generate synthetic samples, even in sparse regions. | ⚠️ Limited to high-density zones.<br>Uses KDE to identify realistic regions, avoiding interpolation in sparse or noisy areas. | SMOTE ensures that all rare classes (e.g., U2R, R2L) are represented.<br>KGSMOTE refines those samples to match realistic distributions. |
| Sample realism                   | ⚠️ May generate noisy or overlapping samples.<br>Interpolation can occur near outliers or in low-density regions, reducing fidelity. | ✅ High realism via KDE-guided sampling.<br>Generates synthetic data only where real samples are densely clustered. | SMOTE provides initial diversity.<br>KGSMOTE enhances quality by anchoring synthetic samples in realistic feature space. |
| Computational cost               | ✅ Lightweight and fast.<br>Scales well for large datasets and quick prototyping. | ⚠️ Computationally intensive.<br>KDE estimation adds overhead, especially for high-dimensional data. | Use SMOTE for initial oversampling.<br>Apply KGSMOTE selectively for precision in critical attack classes. |
| Academic alignment               | ✅ Widely accepted in ML literature.<br>Commonly used in IDS and imbalanced classification research. | ✅ Aligned with KGMS-IDS framework  proposed in [Electronics 2023, 12(9), 3911](https://www.mdpi.com/2079-9292/12/9/3911).<br>Introduces KDE-enhanced realism for minority class synthesis. | Combining both demonstrates methodological rigor.<br>Supports reproducibility and academic credibility. |
| SHAP-based interpretability      | ⚠️ SHAP may misattribute importance due to noisy synthetic samples.<br>Decision boundaries may be distorted. | ✅ KDE-based realism improves SHAP clarity.<br>Feature attributions are more trustworthy and interpretable. | SMOTE expands the decision space.<br>helps the model learn broader decision boundaries.<br>KGSMOTE ensures SHAP values reflect realistic minority class behavior.<br>Together, they helped  IDS pipeline:<br>Detect rare attacks more reliably.<br>Reduce false positives.<br>Improve interpretability.<br>Meet academic benchmarks.<br>which improves SHAP explanations and trustworthiness.

  

## 📁 Files
   - `IDS_Pipeline.ipynb`: End-to-end notebook
   - `SHAP_Explanation_AllRows.xlsx`: Annotated predictions with feature impact
  - `autoencoder_model.h5`: Trained feature reducer
  - `requirements.txt`: Dependencies

### Citation
```markdown

Baher, N. (2025). *Hybrid Oversampling for Intrusion Detection: SMOTE + KGSMOTE*. GitHub Repository: [ids-hybrid-oversampling-smote-kgsmote](https://github.com/Nouribaher/ids-hybrid-oversampling-smote-kgsmote)).

```
