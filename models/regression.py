from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression(df, cv=5, target=None, models=['Linear Regression', 'Random Forest',
                'Decision Tree Regressor', 'Gradient Boosting Regressor'], metrics=['neg_mean_squared_error','neg_mean_absolute_error', 'neg_mean_absolute_percentage_error' 'r2']):

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

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_scores = cross_validate(lr_model, X, y, cv=cv, scoring=metrics)
    lr_scores_mean = {metric: np.mean(lr_scores[f'test_{metric}']) for metric in metrics}
    results['Linear Regression'] = lr_scores_mean

    # Random Forest Model
    rf_model = RandomForestRegressor()
    rf_scores = cross_validate(rf_model, X, y, cv=cv, scoring=metrics)
    rf_scores_mean = {metric: np.mean(rf_scores[f'test_{metric}']) for metric in metrics}
    results['Random Forest'] = rf_scores_mean

    # Extra Model 1: Decision Tree Regressor
    dt_model = DecisionTreeRegressor()
    dt_scores = cross_validate(dt_model, X, y, cv=cv, scoring=metrics)
    dt_scores_mean = {metric: np.mean(dt_scores[f'test_{metric}']) for metric in metrics}
    results['Decision Tree Regressor'] = dt_scores_mean

    # Extra Model 2: Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor()
    gb_scores = cross_validate(gb_model, X, y, cv=cv, scoring=metrics)
    gb_scores_mean = {metric: np.mean(gb_scores[f'test_{metric}']) for metric in metrics}
    results['Gradient Boosting Regressor'] = gb_scores_mean

    results = {"Results" : results}
    return results
