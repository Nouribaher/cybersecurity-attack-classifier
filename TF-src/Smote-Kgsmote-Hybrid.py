def balance(X, y, method='smote', minority_class_id=None, n_samples=500, k_neighbors=2):
    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import KernelDensity
    import pandas as pd
    import numpy as np
    from collections import Counter

    # Filter out classes with too few samples for SMOTE
    def filter_valid_classes(X, y, min_samples):
        counts = Counter(y)
        valid_classes = [cls for cls, count in counts.items() if count >= min_samples]
        mask = np.isin(y, valid_classes)
        return X[mask], y[mask]

    if method == 'smote':
        # Ensure SMOTE won't fail due to rare classes
        X_safe, y_safe = filter_valid_classes(X, y, min_samples=k_neighbors + 1)
        smote = SMOTE(k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_safe, y_safe)
        return X_res, y_res

    elif method == 'kgsmote':
        if minority_class_id is None:
            raise ValueError("You must specify minority_class_id for KGSMOTE.")
        mask = y == minority_class_id
        X_minority = X[mask]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_minority)
        synthetic = kde.sample(n_samples=n_samples)
        X_res = pd.concat([pd.DataFrame(X), pd.DataFrame(synthetic)], ignore_index=True)
        y_res = pd.concat([pd.Series(y), pd.Series([minority_class_id]*n_samples)], ignore_index=True)
        return X_res.values, y_res.values

    elif method == 'hybrid':
        # Filter before SMOTE
        X_safe, y_safe = filter_valid_classes(X, y, min_samples=k_neighbors + 1)
        smote = SMOTE(k_neighbors=k_neighbors)
        X_smote, y_smote = smote.fit_resample(X_safe, y_safe)

        # Apply KGSMOTE to refine one minority class
        if minority_class_id is None:
            raise ValueError("You must specify minority_class_id for hybrid.")
        mask = y_smote == minority_class_id
        X_minority = X_smote[mask]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_minority)
        synthetic = kde.sample(n_samples=n_samples)
        X_final = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(synthetic)], ignore_index=True)
        y_final = pd.concat([pd.Series(y_smote), pd.Series([minority_class_id]*n_samples)], ignore_index=True)
        return X_final.values, y_final.values

    else:
        raise ValueError("Method must be 'smote', 'kgsmote', or 'hybrid'.")
      
