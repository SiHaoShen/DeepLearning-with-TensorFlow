import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(frontsize=14)
    plt.grid(True)

#Trend and Seasonality
def trend(time, slope=0):
    return slope * time

#Create a time series trends upward
time = np.arange(4*365 +1)
baseline = 10
series = trend(time, 0.1)
plt.figure(figsize=(10,6))
plot_series(time, series)
plt.show()

#Create a time series with both trend and seasonality
def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2*np.pi),
                    1/np.exp(3*season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase)%period)/period
    return amplitude * seasonal_pattern(season_time)

baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)
plt.figure(figsize=(10,6))
plot_series(time, series)
plt.show()

slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period = 365, amplitude=amplitude)
plt.figure(figsize=(10,6))
plot_series(time, series)
plt.show()

#Create Noise
def white_noise(time, noise_level =1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) *noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed=42)
plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()

series += noise
plt.figure(figsize=(10,6))
plot_series(time, series)
plt.show()

#Two periods, training and validation period
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    phi1 = 0.5
    phi2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += phi1 * ar[step-50]
        ar[step] += phi2 * ar[step-33]
    return ar[50:] * amplitude

def autocorrelation(time, amplitude, seed = None):
    rnd = np.random.RandomState(seed)
    phi = 0.8
    ar = rnd.randn(lne(time) +1)
    for step in range(1, len(time) + 1):
        ar[step] += phi * ar[step-1]
    return ar[1:] * amplitude

series = autocorrelation(time, 10, seed=42)
plt_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + trend(time, 2)
plt_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
plt_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + seasonality(time, period = 50, amplitude = 150) + trend(time, 2)
series2 = autocorrelation(time, 5, seed=42) + seasonality(time, period = 50, amplitude = 2) + trend(time, -1) + 550
series[200:] = series2[200:]
plot_series(time[:300], series[:300])
plt.show()

def impulses(time, num_impulses, amplitude = 1, seed= None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series

series = impulses(time, 10, seed=42)
plot_series(time, series)
plt.show()

def autocorrelation(source, phis):
    ar = source.copy()
    max_lag = len(phis)
    for step, value in enumerate(source):
        for lag, phi in phis.items():
            if step - lag > 0:
                ar[step] += phi * ar[step-lag]
    return ar

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1:0.99})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1:0.70, 50:0.2})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

series_diff1 = series[1:] - series[:-1]
plot_series(time[1:], series_diff1)

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)