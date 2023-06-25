import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

warnings.filterwarnings('ignore')

def plot_ts(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    mse = mean_squared_error(test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test, y_pred)
    train[int(len(train)*0.9):].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}, MSE: {round(mse,2)}, RMSE: {round(rmse,2)}, MAPE: {round(mape,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

def is_stationary(y):

    # "HO: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
        return True
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
        return False

def smoothing(df, date, target, forecast=60, method="add", seasonal_periods=12):
    if date is not None:
        df = df.set_index(date, drop=True)
    else:
        df = df.set_index(df.columns[0], drop=True)
    train = df[:-forecast]
    test = df[-forecast:]

    alphas = betas = gammas = np.arange(0.20, 1, 0.10)

    abg = list(itertools.product(alphas, betas, gammas))

    def tes_optimizer(train, abg, step=forecast):
        best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
        for comb in abg:
            tes_model = ExponentialSmoothing(train, trend=method, seasonal=method, seasonal_periods=seasonal_periods).\
                fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
            y_pred = tes_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae

        print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
            "best_mae:", round(best_mae, 4))

        return best_alpha, best_beta, best_gamma, best_mae

    best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)


    ############################
    # Final TES Model
    ############################

    final_tes_model = ExponentialSmoothing(train, trend=method, seasonal=method, seasonal_periods=seasonal_periods).\
                fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

    y_pred = final_tes_model.forecast(forecast)

    plot_ts(train, test, y_pred, "Triple Exponential Smoothing")
    
    return y_pred

