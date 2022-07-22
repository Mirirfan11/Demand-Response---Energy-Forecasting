import pandas as pd

"""Load Data"""
path ='\dat_building_PreProcessed_data.xlsx'
df2=pd.read_excel(path,header=1, usecols=[0,1,2,3,4,8,9,10,11,12,13],skiprows=2)
df2.head(2)
"""Alternate way of loading data using Google Colab"""
#from google.colab import files
#uploaded = files.upload()
#df2 = pd.read_csv(io.BytesIO(uploaded['##name_of_file.csv']),header=1,skiprows=2,usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
#df2.head(2)

import tensorflow as tf
print(tf.__version__)


# Commented out IPython magic to ensure Python compatibility.

from pmdarima import auto_arima
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.pylab import rcParams
import numpy as np
from pandas import DataFrame
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.rcParams["figure.figsize"] = (12,6)
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import itertools
import io

df=df2[3876:]    #df.dropna(inplace=True)
df.index=pd.to_datetime(df['Date Time'])   
# df.set_index('Date Time',inplace=True)
# df.index=pd.to_datetime(df.index)
ds=df2[3876:]

"""convert pandas-datetime-to-unix-timestamp-seconds"""

date_time = pd.to_datetime(ds.pop('Date Time'), format='%Y.%m.%d %H:%M:%S')
timestamp_s = date_time.map(pd.Timestamp.timestamp)
Times = (date_time - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

df.head(1)

#Seasonal Decomposition
result=seasonal_decompose(df['VRV_submeter_momentary'][:], model='additive', period=288*7) #model='multiplicative'
fig=result.plot()
fig.set_figheight(4)
plt.show()

plt.figure(figsize=(8, 5))
result.seasonal.plot(legend=('Seasonality'))
plt.show()

plt.figure(figsize=(8, 5))
(result.trend).plot(legend=('Trend'))
plt.show()

"""Autocorrelation and Partial correlation of original series of VRV Submeter momentary"""

autocorrelation_plot(df['VRV_submeter_momentary'].asfreq("5T"))
plt.axis([0, 1500, -0.5, 0.8])
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['VRV_submeter_momentary'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['VRV_submeter_momentary'], lags=40, ax=ax2)
plt.show()

"""#### check the stationarity of the data , Augmented Dickey-Fuller test"""

from statsmodels.tsa.stattools import adfuller    
def test_adf(series, title=''):
    dfout={}
    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key,val in dftest[4].items():
        dfout[f'critical value ({key})']=val
    if dftest[1]<=0.05: # test statistics less than p
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
        print("Data is Stationary", title)
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for", title)

"""If the p-value obtained is greater than significance level of 0.05 and the ADF statistic is higher than any of the critical values. Then, there is no reason to reject the null hypothesis. So, the time series is in fact non-stationary."""

# Commented out IPython magic to ensure Python compatibility.
#example of ADF
from statsmodels.tsa.stattools import adfuller 
series = df['VRV_submeter_momentary'].values   #np.random.randn(100)
result = adfuller(series, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(10,7))
plt.plot(series);
plt.title('VRV_submeter_momentary');

test_adf(df['VRV_submeter_momentary'], title='VRV_submeter_manometery')

"""Autocorrelation of the differenced series of immediate values """

autocorrelation_plot(df['VRV_submeter_momentary'].diff(1)[1:])
plt.axis([0, 50, -0.25, 0.1])

"""Plotting function"""

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

"""Splitting and visualization"""

split_time = int(len(df)*0.775)
time_train = Times[:split_time]
series = df['VRV_submeter_momentary'].values
x_train = series[:split_time]
time_valid = Times[split_time:]
x_valid = series[split_time:]
plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()

"""Naive Forecast"""

series = df['VRV_submeter_momentary'].values     #series = series = df['VRV_submeter_momentary'].asfreq("5T")
naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)

"""Let's zoom in on the start of the validation period:"""

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)

"""You can see that the naive forecast lags 1 step behind the time series.
Now let's compute the mean squared error and the mean absolute error between the forecasts and the predictions in the validation period:
"""

print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

"""That's our baseline, now let's try a moving average:"""

def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)

rw = 72
moving_avg = moving_average_forecast(series, rw)[split_time - rw:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())

"""That's worse than naive forecast! The moving average does not anticipate trend or seasonality, so let's try to remove them by using differencing. Let's say seasonality period is 288 by 7 (weekly seasonality) , we will subtract the value at time t – 288 by 7 from the value at time t.
#### Remove the seasonality factor
"""

shift = 288*7
series_diff1 = series[shift:] - series[:-shift]
diff_time  = Times[shift:]
plot_series(diff_time,series_diff1)
plt.show()

"""Autocorrelation after a single differencing """

#Seasonal Decomposition
result1=seasonal_decompose(series_diff1[:], model='additive', period=288*7) #model='multiplicative'
fig=result1.plot()
fig.set_figheight(4)
plt.show()

"""#### there may be more than one seasonalities 
Data that are observed every 5 minute might have an hourly seasonality (frequency=12), a daily seasonality (frequency=24x12=288), a weekly seasonality (frequency=24x12x7=2016) and an annual seasonality (frequency=24x12x365.25=105192). 
"""

x = diff_time
y = result1.seasonal
z = result1.trend

plt.figure(figsize=(8, 5))
plt.title("seasonality") 
plt.xlabel("Date") 
plt.ylabel("Seasonal") 
plt.plot(x,y) 
plt.show()

plt.figure(figsize=(8, 5))
plt.title("Trend") 
plt.xlabel("Date") 
plt.ylabel("Trend") 
plt.plot(x,z) 
plt.show()

diff_moving_avg = moving_average_forecast(series_diff1, 48)[split_time - shift - 48:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_diff1[split_time - shift:])
plot_series(time_valid, diff_moving_avg)
plt.show()

"""Now let's bring back the trend and seasonality by adding the past values from t – 288:"""

diff_moving_avg_plus_past = series[split_time - shift:-shift] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())

"""Better than first moving av. forecast, good. However the forecasts look a bit too random, because we're just adding past values, which were noisy. Let's use a moving averaging on past values to remove some of the noise:
"""

diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - shift-10:-shift+10], 20) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())

"""# ARMA
Add  new columns of moving average of VRV submeter momentary and Outside Temp to the dataset
"""

import numpy as np
rw = 144
n = rw # change the value accordingly
ds1=df2[3876:]    
MA_VRV_Subeter_momentary = moving_average_forecast(ds1['VRV_submeter_momentary'], n)   
MA_temp_outside  = moving_average_forecast(ds1['outside_Temp'], n)

ds2 = pd.DataFrame({'MA_VRV_Subeter_momentary':MA_VRV_Subeter_momentary,'MA_Temp_Outside':MA_temp_outside})

ds1 = ds1.reset_index()
ds2 = ds2.reset_index()
ds3 = [ds1, ds2]
df_final = pd.concat(ds3, axis=1)
#df_final.to_csv(df_final, index=False)
df_final.index=pd.to_datetime(df_final['Date Time'])
df1 = df_final.drop("index", axis=1)

"""Split data"""

len(X_train)

split_time = int(len(df)*0.775)
X_train = df1[:split_time]
X_valid = df1[split_time:-n]

auto_arima(X_train['MA_VRV_Subeter_momentary'][12000:15000],m=4,trace=True ).summary()

# ARIMA Model
model=sm.tsa.ARIMA(endog=X_train['MA_VRV_Subeter_momentary'],order=(3,0,1))  # Autoregressive with Moving Average
results=model.fit()
print(results.summary())
#results.resid.plot()
#plt.show()
# line plot of residuals
residuals = DataFrame(results.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())


result = results.forecast(288)
plt.plot(result,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:288],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

"""we get a density plot of the residual error values, suggesting the errors are Gaussian, but may not be centered on zero.
A zero mean in residuals suggest no bias in predictions
### ARMA Running After Every 288 predictions/ updating after making each day predictions
"""

m=288
Predicted=pd.Series()
for i in range(0,int(len(X_valid)/m)-5):  
  endotrain=df1[:len(X_train)+i*m]
  Actual_values=X_valid['MA_VRV_Subeter_momentary'][i*m:i*m+m]
  arma_model=sm.tsa.ARIMA(endotrain['MA_VRV_Subeter_momentary'],order=(3,0,0))
  model=arma_model.fit()
  forecast2 = model.forecast(m)
  Predicted=Predicted.append(forecast2) #append to add pd.series to another pd.series
  rmse=sqrt(mean_squared_error(Actual_values,forecast2))
  print('\nRMSE for ' + str(i+1)+' day is:',rmse)


Predicted.index=pd.to_datetime(X_valid['Date Time'][0:m*(i+1)]) #change the index from number to date time
plt.plot(Predicted,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:len(Predicted)],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

plt.rcParams["figure.figsize"] = (12,6)

"""#### ARIMAX"""

df1.head(1)

split_time = int(len(df)*0.80)
X_train = df1[:split_time]
X_valid = df1[split_time:-n]
exoge_train=X_train[['outside_wind','outside_Temp','Groupings_with_holiday']] #exogeneous variables
exoge_test=X_valid[['outside_wind','outside_Temp','Groupings_with_holiday']] #exogeneous variables

arimax_model=sm.tsa.statespace.SARIMAX(X_train['VRV_submeter_momentary'],order=(4,1,1),exog = exoge_train) #,seasonal_order=(1,0,1,7)
model=arimax_model.fit(disp=False)
model.summary()

"""A single Forcasting over the entire timeframe"""

forecast = model.predict(start = len(X_train),end=len(X_train)+len(X_valid)-1, typ='levels',exog=exoge_test).rename("Forecasted VRV Submeter momentary")

X_valid['MA_VRV_Subeter_momentary'].plot(legend=True)
forecast.plot(legend=True)

"""### ARIMAX Running After Every 288 predictions/ updating after making each day predictions"""

m=288
Predicted=pd.Series()
for i in range(0,int(len(X_valid)/m)-7):  #
  exotest=X_valid[['outside_Temp','On_or_Off']][i*m:i*m+m] #change to MA_Temp_Outside
  endotrain=df1[:len(X_train)+i*m]
  exotrain=endotrain[['outside_Temp','On_or_Off']]
  Actual_values=X_valid['MA_VRV_Subeter_momentary'][i*m:i*m+m]
  arimax_model=sm.tsa.statespace.SARIMAX(endotrain['VRV_submeter_momentary'],order=(6,1,1),exog=exotrain)  #change to MA_VRV_Subeter_momentary
  model=arimax_model.fit(disp=False)
  forecast2 = model.forecast(m, exog=exotest)
  Predicted=Predicted.append(forecast2) #append to add pd.series to another pd.series
  rmse=sqrt(mean_squared_error(Actual_values,forecast2))
  print('\nRMSE for ' + str(i+1)+' day is:',rmse)

Predicted.index=pd.to_datetime(X_valid['Date Time'][0:m*(i+1)]) #change the index from number to date time
plt.plot(Predicted,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:len(Predicted)],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

m=288
Predicted=pd.Series()
for i in range(0,int(len(X_valid)/m)-6):  #
  exotest=X_valid[['outside_Temp','On_or_Off']][i*m:i*m+m] #change to MA_Temp_Outside
  endotrain=df1[:len(X_train)+i*m]
  exotrain=endotrain[['outside_Temp','On_or_Off']]
  Actual_values=X_valid['MA_VRV_Subeter_momentary'][i*m:i*m+m]
  arimax_model=sm.tsa.statespace.SARIMAX(endotrain['VRV_submeter_momentary'],order=(2,1,0),exog=exotrain)  #change to MA_VRV_Subeter_momentary
  model=arimax_model.fit(disp=False)
  forecast2 = model.forecast(m, exog=exotest)
  Predicted=Predicted.append(forecast2) #append to add pd.series to another pd.series
  rmse=sqrt(mean_squared_error(Actual_values,forecast2))
  print('\nRMSE for ' + str(i+1)+' day is:',rmse)

Predicted.index=pd.to_datetime(X_valid['Date Time'][0:m*(i+1)]) #change the index from number to date time
plt.plot(Predicted,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:len(Predicted)],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

m=288
Predicted=pd.Series()
for i in range(0,int(len(X_valid)/m)-7):  #
  exotest=X_valid[['outside_Temp','Groupings_with_holiday']][i*m:i*m+m] #change to MA_Temp_Outside
  endotrain=df1[:len(X_train)+i*m]
  exotrain=endotrain[['outside_Temp','Groupings_with_holiday']]
  Actual_values=X_valid['MA_VRV_Subeter_momentary'][i*m:i*m+m]
  arimax_model=sm.tsa.statespace.SARIMAX(endotrain['MA_VRV_Subeter_momentary'],order=(5,1,2),exog=exotrain)  #change to MA_VRV_Subeter_momentary
  model=arimax_model.fit(disp=False)
  forecast2 = model.forecast(m, exog=exotest)
  Predicted=Predicted.append(forecast2) #append to add pd.series to another pd.series
  rmse=sqrt(mean_squared_error(Actual_values,forecast2))
  print('\nRMSE for ' + str(i+1)+' day is:',rmse)

Predicted.index=pd.to_datetime(X_valid['Date Time'][0:m*(i+1)]) #change the index from number to date time
plt.plot(Predicted,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:len(Predicted)],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

m=288
Predicted=pd.Series()
for i in range(0,int(len(X_valid)/m)-7):  #
  exotest=X_valid[['MA_Temp_Outside','Groupings_with_holiday']][i*m:i*m+m] #change to MA_Temp_Outside
  endotrain=df1[:len(X_train)+i*m]
  exotrain=endotrain[['MA_Temp_Outside','Groupings_with_holiday']]
  Actual_values=X_valid['MA_VRV_Subeter_momentary'][i*m:i*m+m]
  arimax_model=sm.tsa.statespace.SARIMAX(endotrain['MA_VRV_Subeter_momentary'],order=(5,1,2),exog=exotrain)  #change to MA_VRV_Subeter_momentary
  model=arimax_model.fit(disp=False)
  forecast2 = model.forecast(m, exog=exotest)
  Predicted=Predicted.append(forecast2) #append to add pd.series to another pd.series
  rmse=sqrt(mean_squared_error(Actual_values,forecast2))
  print('\nRMSE for ' + str(i+1)+' day is:',rmse)

Predicted.index=pd.to_datetime(X_valid['Date Time'][0:m*(i+1)]) #change the index from number to date time
plt.plot(Predicted,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:len(Predicted)],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

m=288
Predicted=pd.Series()
for i in range(0,int(len(X_valid)/m)-7):  #
  exotest=X_valid[['MA_Temp_Outside','On_or_Off']][i*m:i*m+m] #change to MA_Temp_Outside
  endotrain=df1[:len(X_train)+i*m]
  exotrain=endotrain[['MA_Temp_Outside','On_or_Off']]
  Actual_values=X_valid['MA_VRV_Subeter_momentary'][i*m:i*m+m]
  arimax_model=sm.tsa.statespace.SARIMAX(endotrain['MA_VRV_Subeter_momentary'],order=(5,1,2),exog=exotrain)  #change to MA_VRV_Subeter_momentary
  model=arimax_model.fit(disp=False)
  forecast2 = model.forecast(m, exog=exotest)
  Predicted=Predicted.append(forecast2) #append to add pd.series to another pd.series
  rmse=sqrt(mean_squared_error(Actual_values,forecast2))
  print('\nRMSE for ' + str(i+1)+' day is:',rmse)

Predicted.index=pd.to_datetime(X_valid['Date Time'][0:m*(i+1)]) #change the index from number to date time
plt.plot(Predicted,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:len(Predicted)],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

m=288
Predicted=pd.Series()
for i in range(0,int(len(X_valid)/m)-7):  #
  exotest=X_valid[['MA_Temp_Outside','Groups_with_holiday']][i*m:i*m+m] #change to MA_Temp_Outside
  endotrain=df1[:len(X_train)+i*m]
  exotrain=endotrain[['MA_Temp_Outside','Groups_with_holiday']]
  Actual_values=X_valid['MA_VRV_Subeter_momentary'][i*m:i*m+m]
  arimax_model=sm.tsa.statespace.SARIMAX(endotrain['MA_VRV_Subeter_momentary'],order=(5,1,2),exog=exotrain)  #change to MA_VRV_Subeter_momentary
  model=arimax_model.fit(disp=False)
  forecast2 = model.forecast(m, exog=exotest)
  Predicted=Predicted.append(forecast2) #append to add pd.series to another pd.series
  rmse=sqrt(mean_squared_error(Actual_values,forecast2))
  print('\nRMSE for ' + str(i+1)+' day is:',rmse)

Predicted.index=pd.to_datetime(X_valid['Date Time'][0:m*(i+1)]) #change the index from number to date time
plt.plot(Predicted,label='Predicted')
plt.plot(X_valid['MA_VRV_Subeter_momentary'][:len(Predicted)],label='Actual')
plt.xlabel('Date')
plt.ylabel('MA_VRV_Subeter_momentary')
plt.title('Predicted Vs Actual')
plt.legend()
plt.show()

split_time = int(len(df)*0.83)
X_train = df1[:split_time]
X_valid = df1[split_time:-n]

X_train['MA_VRV_Subeter_momentary'].plot()



"""#### Normalization"""

#Normalization
from sklearn.preprocessing import MinMaxScaler    #Normalization of data between (0,1)
sc_in = MinMaxScaler(feature_range=(0, 1))
dq = sc_in.fit_transform(df[['VRV_submeter_momentary','outside_Temp','outside_wind']])
scaled_input =pd.DataFrame(dq)
X= scaled_input
X=X.rename(columns={0:'VRV_submeter_momentary',1:'outside_Temp',2:'outside_wind'})
X.index=pd.to_datetime(df['Date Time'])
X.tail()

"""#### DARTS"""

!pip install darts

# Commented out IPython magic to ensure Python compatibility.
from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FFT
)

from darts.metrics import mape, mase
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.datasets import AirPassengersDataset

from darts.dataprocessing.transformers import Scaler, MissingValuesFiller, Mapper, InvertibleMapper
from darts.dataprocessing import Pipeline

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import reduce

from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FFT
)

from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.datasets import AirPassengersDataset
from darts.dataprocessing.transformers import Scaler
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

from darts.utils.timeseries_generation import gaussian_timeseries, linear_timeseries, sine_timeseries
from darts.models import RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel
from darts.metrics import mape, smape,rmse
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
import torch   
import numpy as np

torch.manual_seed(1); np.random.seed(1)  # for reproducibility

from darts import TimeSeries
from darts.utils.timeseries_generation import gaussian_timeseries, linear_timeseries, sine_timeseries
from darts.models import RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel
from darts.metrics import mape, smape, rmse
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset

torch.manual_seed(1); np.random.seed(1)  # for reproducibility

series_endog = TimeSeries.from_dataframe(df1, 'Date Time', 'VRV_submeter_momentary') #changing into the data dart can read
series_endog_MA = TimeSeries.from_dataframe(df1, 'Date Time', 'MA_VRV_Subeter_momentary') #changing into the data dart can read
series_On_or_Off= TimeSeries.from_dataframe(df1, 'Date Time', 'On_or_Off')
series_Groups_with_holiday = TimeSeries.from_dataframe(df1, 'Date Time', 'Groups_with_holiday')
series_Groupings_with_holiday	 = TimeSeries.from_dataframe(df1, 'Date Time', 'Groupings_with_holiday')
series_temp_out = TimeSeries.from_dataframe(df1, 'Date Time', 'outside_Temp')
series_MA_Temp_Outside = TimeSeries.from_dataframe(df1, 'Date Time', 'MA_Temp_Outside')

#series_endog

"""#### Endogeneous variable is Averaged VRV Submeter Momentary with exogeneous variable as T"""

series_endog_MA.plot(label='MA_VRV_Subeter_momentary')
series_Groups_with_holiday.plot(label='Groups_with_holidays')
plt.legend();

scaler_endog, scaler_exog = Scaler(), Scaler()
series_endog_scaled = scaler_endog.fit_transform(series_endog_MA) # endog = MA_VRV_Subeter_momentary,
series_exog_scaled=scaler_exog.fit_transform(series_Groups_with_holiday)

series_endog_scaled.plot(label='MA_VRV_Subeter_momentary')
series_exog_scaled.plot(label='Groups_with_holiday')
plt.legend();

train_endog, val_endog = series_endog_scaled[:int(0.9*len(df1))], series_endog_scaled[int(0.9*len(df1)):]
train_exog, val_exog = series_exog_scaled[:int(0.9*len(df1))], series_exog_scaled[int(0.9*len(df1)):]

len(train_exog),len(val_endog)

from darts.models import RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel
model_endog = NBEATSModel(input_chunk_length=800, output_chunk_length=288, n_epochs=150, random_state=0)

model_endog.fit(train_endog, verbose=True)

pred = model_endog.predict(n=len(val_endog))  #change it to m number of days (n=m*288)

val_endog.plot(label='Actual') # only values of one day
pred.plot(label='forecast')
plt.legend();
plt.title('N days forecast')
print('MAPE = {:.2f}%'.format(mape(series_endog_scaled, pred)))

days=1 # change days according to the number of days to forecast
pred = model_endog.predict(n=days*288)  
predicted_Inv = scaler_endog.inverse_transform(pred) # to transform series back to the original
Actual_endog_Inv = scaler_endog.inverse_transform(val_endog)  # to transform scaled validation series back to original values
Actual_endog_Inv[0:days*288].plot(label='Actual')
predicted_Inv[0:days*288].plot(label='forecast')  # plot forecasting for required number of days 
plt.legend();
plt.title('2 Day Forecasting')
print('RMSE = {:.2f}%'.format(rmse(Actual_endog_Inv[0:days*288], predicted_Inv[0:days*288])))



"""Endogeneous variable is Averaged VRV Submeter Momentary with exogeneous variable as T"""

series_endog_MA.plot(label='MA_VRV_Subeter_momentary')
series_Groupings_with_holiday.plot(label='Groupings with holidays')
plt.legend();

scaler_endog, scaler_exog = Scaler(), Scaler()
series_endog_scaled = scaler_endog.fit_transform(series_endog_MA) # endog = MA_VRV_Subeter_momentary,
series_exog_scaled=scaler_exog.fit_transform(series_Groupings_with_holiday)     #exog = On_or_Off

series_endog_scaled.plot(label='MA_VRV_Subeter_momentary')
series_exog_scaled.plot(label='Groupings with holiday')
plt.legend();

train_endog, val_endog = series_endog_scaled[:int(0.9*len(df1))], series_endog_scaled[int(0.9*len(df1)):]
train_exog, val_exog = series_exog_scaled[:int(0.9*len(df1))], series_exog_scaled[int(0.9*len(df1)):]



from darts.models import RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel
model_endog = NBEATSModel(input_chunk_length=500, output_chunk_length=288, n_epochs=24, random_state=0)

model_endog.fit(train_endog, verbose=True)

pred = model_endog.predict(n=len(val_endog))  #change it to m number of days (n=m*288)

val_endog.plot(label='actual') # only values of one day
pred.plot(label='forecast')
plt.legend();
plt.title('N days forecast')
print('MAPE = {:.2f}%'.format(mape(series_endog_scaled, pred)))

days=2 # change days according to the number of days to forecast
pred = model_endog.predict(n=days*288)  
predicted_Inv = scaler_endog.inverse_transform(pred) # to transform series back to the original
Actual_endog_Inv = scaler_endog.inverse_transform(val_endog)  # to transform scaled validation series back to original values
Actual_endog_Inv[0:days*288].plot(label='Actual')
predicted_Inv[0:days*288].plot(label='forecast')  # plot forecasting for required number of days 
plt.legend();
plt.title('2 Day Forecasting')
print('RMSE = {:.2f}%'.format(rmse(Actual_endog_Inv[0:days*288], predicted_Inv[0:days*288])))



"""# NARX AND DIRECT AUTO REGRESSOR"""

!pip install fireTS

split_time = int(len(df1)*0.83)
X_train = df1[:split_time]
X_valid = df1[split_time:-n]
Xtrain=X_train[['MA_Temp_Outside','On_or_Off']] #exogeneous variables
ytrain = X_train['MA_VRV_Subeter_momentary'] #endogeneous variables
Xtest=X_valid[['MA_Temp_Outside','On_or_Off']] 
ytest = X_valid['MA_VRV_Subeter_momentary']

"""## Build the NARX model"""

narx_mdl = NARX(LinearRegression(), auto_order=6, exog_order=[3, 3], exog_delay=[0, 0])
