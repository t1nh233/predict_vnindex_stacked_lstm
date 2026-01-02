# =====================================================
# ARIMA FORECASTING – VN INDEX (FULL WORKING VERSION)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# 1. LOAD & PREPROCESS DATA
# =====================================================
file_path = "vn_index_historical_data_9_12.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

df['Price'] = df['Price'].str.replace(',', '').astype(float)
df.rename(columns={'Price': 'Close'}, inplace=True)

# set business day frequency & fill missing days
df = df.asfreq('B')
df['Close'] = df['Close'].ffill()

# log transform (recommended for financial series)
close_series = np.log(df['Close'])

print(close_series.info())

# =====================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# =====================================================
total_len = len(close_series)

train_size = int(total_len * 0.7)
val_size   = int(total_len * 0.2)

train_data = close_series[:train_size]
val_data   = close_series[train_size:train_size + val_size]
test_data  = close_series[train_size + val_size:]

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

# =====================================================
# 3. STATIONARITY TEST (ADF)
# =====================================================
adf_raw = adfuller(train_data.dropna())
print("\nADF raw p-value:", adf_raw[1])

diff_train = train_data.diff().dropna()
adf_diff = adfuller(diff_train)
print("ADF diff p-value:", adf_diff[1])

# =====================================================
# 4. ACF & PACF
# =====================================================
plot_acf(diff_train, lags=30)
plt.title("ACF of Differenced Series")
plt.show()

plot_pacf(diff_train, lags=20, method='ywm')
plt.title("PACF of Differenced Series")
plt.show()

# =====================================================
# 5. GRID SEARCH (p, q)
# =====================================================
best_rmse = np.inf
best_order = None

print("\nGrid search ARIMA(p,1,q):")
for p in [0, 1, 2, 3]:
    for q in [0, 1, 2]:
        try:
            model = ARIMA(train_data, order=(p, 1, q))
            model_fit = model.fit()

            val_forecast = model_fit.forecast(steps=len(val_data))
            rmse = np.sqrt(mean_squared_error(val_data, val_forecast))

            print(f"ARIMA({p},1,{q}) -> RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p, 1, q)

        except Exception as e:
            print(f"ARIMA({p},1,{q}) failed")

print("\nBest ARIMA order:", best_order)

# =====================================================
# 6. ROLLING FORECAST (SLIDING WINDOW)
# =====================================================
WINDOW = 250
history = list(pd.concat([train_data, val_data]).values)

rolling_forecast = []

for t in range(len(test_data)):
    window_data = history[-WINDOW:]

    model = ARIMA(window_data, order=best_order)
    model_fit = model.fit()

    yhat = model_fit.forecast()[0]
    rolling_forecast.append(yhat)

    history.append(test_data.iloc[t])

rolling_forecast = pd.Series(rolling_forecast, index=test_data.index)

# =====================================================
# 7. EVALUATION (LOG SCALE)
# =====================================================
mae = mean_absolute_error(test_data, rolling_forecast)
rmse = np.sqrt(mean_squared_error(test_data, rolling_forecast))
r2 = r2_score(test_data, rolling_forecast)

print("\n=== TEST PERFORMANCE (LOG SCALE) ===")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# =====================================================
# 8. RESIDUAL DIAGNOSTICS
# =====================================================
residuals = test_data - rolling_forecast

lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("\nLjung-Box test:")
print(lb_test)

# =====================================================
# 9. PLOT (BACK TO ORIGINAL SCALE)
# =====================================================
plt.figure(figsize=(12,6))
plt.plot(np.exp(test_data), label='Actual VN-Index', color='blue')
plt.plot(np.exp(rolling_forecast), label='ARIMA Forecast', color='red')
plt.title('VN-Index Forecast using ARIMA (Rolling)')
plt.xlabel('Date')
plt.ylabel('VN-Index')
plt.legend()
plt.grid(True)
plt.show()
