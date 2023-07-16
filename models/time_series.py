import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def time_series(df, target=None, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), method='powell', forecast=10):

    if len(df.columns) > 2:
        exogenous_columns = list(df.columns.drop(target))  # Exogenous columns
        for column in exogenous_columns:
            if df[column].dtype == 'datetime64[ns]':
                exogenous_columns.remove(column)
        df = df.dropna()

        # SARIMAX Model
        sarimax_model = SARIMAX(df[target], exog=df[exogenous_columns], order=order, seasonal_order=seasonal_order)
        sarimax_model_fit = sarimax_model.fit(method=method)
        exog_forecast = df[exogenous_columns][-(forecast+1):]  # Exogenous data for the last 11 periods
        sarimax_pred = sarimax_model_fit.predict(start=len(df), end=len(df) + forecast, exog=exog_forecast)  # Example forecast
        sarimax_mae = mean_absolute_error(df[target][-(forecast+1):], sarimax_pred)
        sarimax_mse = mean_squared_error(df[target][-(forecast+1):], sarimax_pred)
        sarimax_rmse = np.sqrt(sarimax_mse)
        sarimax_mape = mean_absolute_percentage_error(df[target][-(forecast+1):], sarimax_pred)
        print('SARIMAX Model - MAE:', sarimax_mae)
        print('SARIMAX Model - MSE:', sarimax_mse)
        print('SARIMAX Model - RMSE:', sarimax_rmse)
        print('SARIMAX Model - MAPE:', sarimax_mape)

        metrics = {
            'SARIMAX': (sarimax_mae, sarimax_mse, sarimax_rmse, sarimax_mape)
        }

    else:
        # SARIMA Model
        sarima_model = SARIMAX(df[target], order=order, seasonal_order=seasonal_order)
        sarima_model_fit = sarima_model.fit(method=method)
        sarima_pred = sarima_model_fit.predict(start=len(df), end=len(df) + forecast)  # Example forecast
        sarima_mae = mean_absolute_error(df[target][-(forecast+1):], sarima_pred)
        sarima_mse = mean_squared_error(df[target][-(forecast+1):], sarima_pred)
        sarima_rmse = np.sqrt(sarima_mse)
        sarima_mape = mean_absolute_percentage_error(df[target][-(forecast+1):], sarima_pred)
        print('SARIMA Model - MAE:', sarima_mae)
        print('SARIMA Model - MSE:', sarima_mse)
        print('SARIMA Model - RMSE:', sarima_rmse)
        print('SARIMA Model - MAPE:', sarima_mape)

        # ARIMA Model
        arima_model = ARIMA(df[target], order=order)
        arima_model_fit = arima_model.fit()
        arima_pred = arima_model_fit.predict(start=len(df), end=len(df) + forecast)  # Example forecast
        arima_mae = mean_absolute_error(df[target][-(forecast+1):], arima_pred)
        arima_mse = mean_squared_error(df[target][-(forecast+1):], arima_pred)
        arima_rmse = np.sqrt(arima_mse)
        arima_mape = mean_absolute_percentage_error(df[target][-(forecast+1):], arima_pred)
        print('ARIMA Model - MAE:', arima_mae)
        print('ARIMA Model - MSE:', arima_mse)
        print('ARIMA Model - RMSE:', arima_rmse)
        print('ARIMA Model - MAPE:', arima_mape)

        """
        # Exponential Smoothing Model
        exp_smoothing_model = ExponentialSmoothing(df[target])
        exp_smoothing_model_fit = exp_smoothing_model.fit()
        exp_smoothing_pred = exp_smoothing_model_fit.predict(start=len(df), end=len(df) + forecast)  # Example forecast
        exp_smoothing_mae = mean_absolute_error(df[target][-(forecast+1):], exp_smoothing_pred)
        exp_smoothing_mse = mean_squared_error(df[target][-(forecast+1):], exp_smoothing_pred)
        exp_smoothing_rmse = np.sqrt(exp_smoothing_mse)
        exp_smoothing_mape = mean_absolute_percentage_error(df[target][-(forecast+1):], exp_smoothing_pred)
        print('Exponential Smoothing Model - MAE:', exp_smoothing_mae)
        print('Exponential Smoothing Model - MSE:', exp_smoothing_mse)
        print('Exponential Smoothing Model - RMSE:', exp_smoothing_rmse)
        print('Exponential Smoothing Model - MAPE:', exp_smoothing_mape)
        """

        metrics = {
            'SARIMA': (sarima_mae, sarima_mse, sarima_rmse, sarima_mape),
            'ARIMA': (arima_mae, arima_mse, arima_rmse, arima_mape)
            # 'Exponential Smoothing': (exp_smoothing_mae, exp_smoothing_mse, exp_smoothing_rmse, exp_smoothing_mape)
        }

        def plot_ts(train, test, y_pred, title):
            mae = mean_absolute_error(test, y_pred)
            mse = mean_squared_error(test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(test, y_pred)
            train[int(len(train)*0.9):].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}, MSE: {round(mse,2)}, RMSE: {round(rmse,2)}, MAPE: {round(mape,2)}")
            test.plot(legend=True, label="TEST", figsize=(6, 4))
            y_pred.plot(legend=True, label="PREDICTION")
            plt.show()
            
        # Calculate normalized metrics
        normalized_metrics = {}
        for model, metric_values in metrics.items():
            normalized_values = [metric / max(metric_values) for metric in metric_values]
            normalized_metrics[model] = normalized_values

        # Calculate average of normalized metrics
        average_metrics = {}
        for model, metric_values in normalized_metrics.items():
            average_metric = sum(metric_values) / len(metric_values)
            average_metrics[model] = average_metric

        # Find best model with minimum average metric
        best_model = min(average_metrics, key=average_metrics.get)

        # Print normalized and average metrics
        for model, metric_values in normalized_metrics.items():
            print(model, 'Normalized Metrics:', metric_values)
            print(model, 'Average Metric:', average_metrics[model])

        # Print best model
        return 'Best Model:', best_model