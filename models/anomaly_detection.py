import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def calculate_pca(df, comp_ratio=0.95, target=None):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() < 20:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
    df.fillna(df.mean(), inplace=True)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    pca = PCA()
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    if comp_ratio <= 1:
        n_components = np.argmax(cumsum_var_ratio >= comp_ratio) + 1
    else:
        n_components = int(comp_ratio)
    n_components = min(n_components, 6)  # Limit the number of components to 6 if it exceeds that number
    pca = PCA(n_components=n_components)
    pca.fit(df[numeric_cols])
    transformed_df = pd.DataFrame(pca.transform(df[numeric_cols]), columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    transformed_df = transformed_df.round(decimals=2)  # Round the values to two decimal places
    return transformed_df


def anomaly_detection(df, target=None, cv=5,models = ["IsolationForest", "OneClassSVM", "EllipticEnvelope"], metrics=['accuracy', 'precision_macro', 'f1_macro'],comp_ratio=0.95):
    print('Processing.........')
    
    # Hedef değişkendeki tüm 1'leri içeren satırları al
    positive_samples = df[df[target] == 1]
    
    # Veri setini 3000 satır ile sınırla
    negative_samples = df[df[target] == 0].sample(n=3000-len(positive_samples), random_state=42)
    df = pd.concat([positive_samples, negative_samples], ignore_index=True)
    
    y = df[target]  # Hedef değişkeni
    
    # PCA uygula
    X = calculate_pca(df, comp_ratio=comp_ratio, target=target)
    
    
    results = {}
    print('Model loading 1.......')
    # Isolation Forest modelini oluştur
    isolation_model = IsolationForest()
    isolation_scores = cross_validate(isolation_model, X, y, cv=cv, scoring=metrics)
    isolation_scores_mean = {metric: np.mean(isolation_scores[f'test_{metric}']) for metric in metrics}
    results['IsolationForest'] = isolation_scores_mean

    print('Model loading 2.......')
    svm_model = OneClassSVM()
    oneclasssvm_scores = cross_validate(svm_model, X, y, cv=cv, scoring=metrics)
    svm_scores_mean = {metric: np.mean(oneclasssvm_scores[f'test_{metric}']) for metric in metrics}
    results['OneClassSVM'] = svm_scores_mean

    print('Model loading 3.......')
    elliptic_model = EllipticEnvelope()
    elliptic_scores = cross_validate(elliptic_model, X, y, cv=cv, scoring=metrics)
    elliptic_scores_mean = {metric: np.mean(elliptic_scores[f'test_{metric}']) for metric in metrics}
    results['EllipticEnvelope'] = elliptic_scores_mean
    
    results = {"Results" : results}
    return results
