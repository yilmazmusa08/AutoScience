import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def plot_ts(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    mse = mean_squared_error(test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test, y_pred)
    train[int(len(train)*0.9):].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}, MSE: {round(mse,2)}, RMSE: {round(rmse,2)}, MAPE: {round(mape,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

def sarimax_optimize(df, date, target, forecast=60):

    if date is not None:
        df = df.set_index(date, drop=True)
    else:
        df = df.set_index(df.columns[0], drop=True)
    
    df = df[target]
    print(df)
    
    train = df[:-forecast] 
    test = df[-forecast:] 

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    def sarima_optimizer_mae(train, pdq, seasonal_pdq):
        best_mae, best_order, best_seasonal_order = float("inf"), None, None
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                    sarima_model = model.fit(disp=0)
                    y_pred_test = sarima_model.get_forecast(steps=forecast)
                    y_pred = y_pred_test.predicted_mean
                    mae = mean_absolute_error(test, y_pred)
                    if mae < best_mae:
                        best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                    print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
                except:
                    continue
        print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
        return best_order, best_seasonal_order

    best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

    model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
    sarima_final_model = model.fit(disp=0)

    y_pred_test = sarima_final_model.get_forecast(steps=forecast)
    y_pred = y_pred_test.predicted_mean
    y_pred = pd.Series(y_pred, index=test.index)

    plot_ts(train, test, y_pred, "SARIMA")



    ############################
    # Final Model
    ############################

    model = SARIMAX(df, order=best_order, seasonal_order=best_seasonal_order)
    sarima_final_model = model.fit(disp=0)

    feature_predict = sarima_final_model.get_forecast(steps=6)
    feature_predict = feature_predict.predicted_mean

    print(feature_predict)

    return feature_predict