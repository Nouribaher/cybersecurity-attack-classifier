# Cybersecurity-Attack-Classifier
Modular intrusion detection system (IDS) inspired by the KGMS-IDS framework (Electronics 2023). Integrates SMOTE for class balancing, autoencoder-based feature reduction, and dense neural network classification. Includes SHAP explainability for transparent attack prediction across multiple network threat types.
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

![Pipeline Diagram](docs/pipeline.png)

## 📊 Results
- Accuracy: 98.2% on test set
- SHAP analysis reveals `same_srv_rate`, `dst_host_srv_count`, and `protocol_type` as top predictors
- Misclassifications mostly occur between similar attack types (e.g., Probe vs DoS)

## 📁 Files
- `IDS_Pipeline.ipynb`: End-to-end notebook
- `SHAP_Explanation_AllRows.xlsx`: Annotated predictions with feature impact
- `autoencoder_model.h5`: Trained feature reducer
- `requirements.txt`: Dependencies

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/main.py
