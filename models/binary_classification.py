import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def binary_classification(df, cv=5, target=None, models=['Logistic Regression', 'Random Forest',
                'Decision Tree Classifier', 'Gradient Boosting Classifier',
                'Naive Bayes', 'Support Vector Machines', 'AdaBoost', 'XGBoost',
                'LightGBM'], metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):

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

    results = {}

    for model in models:
        if model == "Logistic Regression":
            # Logistic Regression Model
            lr_model = LogisticRegression()
            lr_scores = cross_validate(lr_model, X, y, cv=cv, scoring=metrics)
            lr_scores_mean = {metric: round(np.mean(lr_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Logistic Regression'] = lr_scores_mean

        elif model == "Random Forest":
            # Random Forest Model
            rf_model = RandomForestClassifier()
            rf_scores = cross_validate(rf_model, X, y, cv=cv, scoring=metrics)
            rf_scores_mean = {metric: round(np.mean(rf_scores[f'test_{metric}']), 2)for metric in metrics}
            results['Random Forest'] = rf_scores_mean

        elif model == "Decision Tree Classifier":
            # Extra Model 1: Decision Tree Classifier
            dt_model = DecisionTreeClassifier()
            dt_scores = cross_validate(dt_model, X, y, cv=cv, scoring=metrics)
            dt_scores_mean = {metric: round(np.mean(dt_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Decision Tree Classifier'] = dt_scores_mean

        elif model == "Gradient Boosting Classifier":
            # Extra Model 2: Gradient Boosting Classifier
            gb_model = GradientBoostingClassifier()
            gb_scores = cross_validate(gb_model, X, y, cv=cv, scoring=metrics)
            gb_scores_mean = {metric: round(np.mean(gb_scores[f'test_{metric}']), 2)for metric in metrics}
            results['Gradient Boosting Classifier'] = gb_scores_mean

        elif model == "Naive Bayes":
            # Extra Model 3: Naive Bayes
            nb_model = GaussianNB()
            nb_scores = cross_validate(nb_model, X, y, cv=cv, scoring=metrics)
            nb_scores_mean = {metric: round(np.mean(nb_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Naive Bayes'] = nb_scores_mean

        elif model == "Support Vector Machines":
            # Extra Model 4: Support Vector Machines
            svm_model = SVC()
            svm_scores = cross_validate(svm_model, X, y, cv=cv, scoring=metrics)
            svm_scores_mean = {metric: round(np.mean(svm_scores[f'test_{metric}']), 2)for metric in metrics}
            results['Support Vector Machines'] = svm_scores_mean

        elif model == "AdaBoost":
            # Extra Model 5: AdaBoost
            ab_model = AdaBoostClassifier()
            ab_scores = cross_validate(ab_model, X, y, cv=cv, scoring=metrics)
            ab_scores_mean = {metric: round(np.mean(ab_scores[f'test_{metric}']), 2) for metric in metrics}
            results['AdaBoost'] = ab_scores_mean

        elif model == "XGBoost":
            # Extra Model 6: XGBoost
            xgb_model = XGBClassifier()
            xgb_scores = cross_validate(xgb_model, X, y, cv=cv, scoring=metrics)
            xgb_scores_mean = {metric: round(np.mean(xgb_scores[f'test_{metric}']), 2) for metric in metrics}
            results['XGBoost'] = xgb_scores_mean

        elif model == "LightGBM":
            # Extra Model 7: LightGBM
            lgbm_model = LGBMClassifier()
            lgbm_scores = cross_validate(lgbm_model, X, y, cv=cv, scoring=metrics)
            lgbm_scores_mean = {metric: round(np.mean(lgbm_scores[f'test_{metric}']), 2) for metric in metrics}
            results['LightGBM'] = lgbm_scores_mean



    results = {"Results": results}
    return results

