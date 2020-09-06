import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

#Set up the time series with seasonality, trend and noise
def plot_series(time, series, format="-", start = 0, end = None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    #Arbitary pattern
    return np.where(season_time < 0.4,
                    np.cos(season_time*2*np.pi),
                    1/np.exp(3*season_time))

def seasonality(time, period, amplitude=1, phase=0):
    #Repeat the pattern
    season_time = ((time + phase) % period)/ period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4*365 + 1, dtype ="float 32")
baseline = 10
series = trend(time, 0.1)
amplitude = 40
slope = 0.05
noise_level = 5

#Create the series
series = baseline + trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude)
#Update with noise
series += noise(time, noise_level, seed=42)
plt.figure(figsize=(10,6))
plot_series(time, series)
plt.show()

#Start Forecasting
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
plt.figure(figsize=(10,6))
plot_series(time_train, x_train)
plt.show()
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid)
plt.show()

#Naive Forecast
naive_forecast = series[split_time-1:-1]
plt.figure(time_valid, x_valid)
plot_series(time_valid, x_valid)
plot_series(time, naive_forecast)
plot_series(time_valid, x_valid, start=0, end=150) #Zoom in
plot_series(time_valid, naive_forecast, start=1, end=151)

#Mean squared error and mean absolute error as baseline
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy()) #61.827
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()) #5.938

#Moving Average
def moving_average_forecast(series, window_size):
    #If window_size = 1, then this is equivalent to naive forecast
    forcast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time -30:]
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy()) #106.674
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy()) #7.142

#Using differencing
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]
plt.figure(figsize=(10,6))
plot_series(diff_time, diff_series)
plt.show()

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time-365-50:]
plt.figure(figsize=(10,6))
plot_series(time_valid, diff_series[split_time -365:])
plot_series(time_valid, diff_moving_avg)
plt.show()

#Bring Back the trend and seasonality
diff_moving_avg_plus_past = series[splt_time-365:-365] + diff_moving_avg
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy()) #52.973
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy()) #5.839

#Use a moving averaging on past values to remove some of the noise
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()) #33.452
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()) #4.569