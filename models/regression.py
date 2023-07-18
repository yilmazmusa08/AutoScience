from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def regression(df, cv=5, target=None, models=['Linear Regression', 'Random Forest', 'Decision Tree Regressor', 'Gradient Boosting Regressor',
                                              'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression', 'Polynomial Regression',
                                              'Support Vector Regression', 'XGBoost Regression', 'LightGBM Regression', 'CatBoost Regression'],
                metrics=['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'r2']):
    
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

    for model in models:
        if model == "Linear Regression":
            # Linear Regression Model
            lr_model = LinearRegression()
            lr_scores = cross_validate(lr_model, X, y, cv=cv, scoring=metrics)
            lr_scores_mean = {metric: round(np.mean(lr_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Linear Regression'] = lr_scores_mean

        elif model == "Random Forest":
            # Random Forest Model
            rf_model = RandomForestRegressor()
            rf_scores = cross_validate(rf_model, X, y, cv=cv, scoring=metrics)
            rf_scores_mean = {metric: round(np.mean(rf_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Random Forest'] = rf_scores_mean

        elif model == "Decision Tree Regressor":
            # Decision Tree Regressor
            dt_model = DecisionTreeRegressor()
            dt_scores = cross_validate(dt_model, X, y, cv=cv, scoring=metrics)
            dt_scores_mean = {metric: round(np.mean(dt_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Decision Tree Regressor'] = dt_scores_mean

        elif model == "Gradient Boosting Regressor":
            # Gradient Boosting Regressor
            gb_model = GradientBoostingRegressor()
            gb_scores = cross_validate(gb_model, X, y, cv=cv, scoring=metrics)
            gb_scores_mean = {metric: round(np.mean(gb_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Gradient Boosting Regressor'] = gb_scores_mean

        elif model == "Ridge Regression":
            # Ridge Regression
            ridge_model = Ridge()
            ridge_scores = cross_validate(ridge_model, X, y, cv=cv, scoring=metrics)
            ridge_scores_mean = {metric: round(np.mean(ridge_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Ridge Regression'] = ridge_scores_mean

        elif model == "Lasso Regression":
            # Lasso Regression
            lasso_model = Lasso()
            lasso_scores = cross_validate(lasso_model, X, y, cv=cv, scoring=metrics)
            lasso_scores_mean = {metric: round(np.mean(lasso_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Lasso Regression'] = lasso_scores_mean

        elif model == "Elastic Net Regression":
            # Elastic Net Regression
            elasticnet_model = ElasticNet()
            elasticnet_scores = cross_validate(elasticnet_model, X, y, cv=cv, scoring=metrics)
            elasticnet_scores_mean = {metric: round(np.mean(elasticnet_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Elastic Net Regression'] = elasticnet_scores_mean

        elif model == "Polynomial Regression":
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            # Linear regression model with polynomial features
            lr_model = LinearRegression()
            poly_scores = cross_validate(lr_model, X_poly, y, cv=cv, scoring=metrics)
            poly_scores_mean = {metric: round(np.mean(poly_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Polynomial Regression'] = poly_scores_mean

        elif model == "Support Vector Regression":
            # Support Vector Regression
            svr_model = SVR()
            svr_scores = cross_validate(svr_model, X, y, cv=cv, scoring=metrics)
            svr_scores_mean = {metric: round(np.mean(svr_scores[f'test_{metric}']), 2) for metric in metrics}
            results['Support Vector Regression'] = svr_scores_mean

        elif model == "XGBoost Regression":
            # XGBoost Regression
            xgb_model = XGBRegressor()
            xgb_scores = cross_validate(xgb_model, X, y, cv=cv, scoring=metrics)
            xgb_scores_mean = {metric: round(np.mean(xgb_scores[f'test_{metric}']), 2) for metric in metrics}
            results['XGBoost Regression'] = xgb_scores_mean

        elif model == "LightGBM Regression":
            # LightGBM Regression
            lgbm_model = LGBMRegressor()
            lgbm_scores = cross_validate(lgbm_model, X, y, cv=cv, scoring=metrics)
            lgbm_scores_mean = {metric: round(np.mean(lgbm_scores[f'test_{metric}']), 2) for metric in metrics}
            results['LightGBM Regression'] = lgbm_scores_mean

        elif model == "CatBoost Regression":
            # CatBoost Regression
            catboost_model = CatBoostRegressor()
            catboost_scores = cross_validate(catboost_model, X, y, cv=cv, scoring=metrics)
            catboost_scores_mean = {metric: round(np.mean(catboost_scores[f'test_{metric}']), 2) for metric in metrics}
            results['CatBoost Regression'] = catboost_scores_mean

    results = {"Results": results}
    return results



