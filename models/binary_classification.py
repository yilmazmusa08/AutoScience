from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import numpy as np


def binary_classification(df, cv=5, target=None, models=['Logistic Regression', 'Random Forest',
                'Decision Tree Classifier', 'Gradient Boosting Classifier'], metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):

    try:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(float)
    except:
        pass

    df = df.dropna()
    y = df[target]
    df = df.select_dtypes(include='number')
    X = df.drop(target, axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    results = {}

    # Logistic Regression Model
    lr_model = LogisticRegression()
    lr_scores = cross_validate(lr_model, X, y, cv=cv, scoring=metrics)
    lr_scores_mean = {metric: np.mean(lr_scores[f'test_{metric}']) for metric in metrics}
    results['Logistic Regression'] = lr_scores_mean

    # Random Forest Model
    rf_model = RandomForestClassifier()
    rf_scores = cross_validate(rf_model, X, y, cv=cv, scoring=metrics)
    rf_scores_mean = {metric: np.mean(rf_scores[f'test_{metric}']) for metric in metrics}
    results['Random Forest'] = rf_scores_mean

    # Extra Model 1: Decision Tree Classifier
    dt_model = DecisionTreeClassifier()
    dt_scores = cross_validate(dt_model, X, y, cv=cv, scoring=metrics)
    dt_scores_mean = {metric: np.mean(dt_scores[f'test_{metric}']) for metric in metrics}
    results['Decision Tree Classifier'] = dt_scores_mean

    # Extra Model 2: Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier()
    gb_scores = cross_validate(gb_model, X, y, cv=cv, scoring=metrics)
    gb_scores_mean = {metric: np.mean(gb_scores[f'test_{metric}']) for metric in metrics}
    results['Gradient Boosting Classifier'] = gb_scores_mean
    
    results = {"Results" : results}
    return results

