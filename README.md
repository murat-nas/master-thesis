# master-thesis
DATA SETS

The data used in this work is IBB Hourly Traffic Density Data covering the time span of January 2020 and September 2023. The values are taken from three sensors.


In this study, a 12-month data file having data from three sensors for year 2022 is used.
The location informations of the selected sensors are as follows:

Sensor1 sxk96j :
Latitude: 41.03118896° North Parallel Longitude: 28.92150879° East Meridian

Sensor2 sxk990 :
Latitude: 41.04766846° North Parallel Longitude: 28.87756348° Eastern Meridian

Sensor3 sxk9gd :
Latitude: 41.1026001° North Parallel Longitude:      28.98742676° Eastern Meridian

Time series that will be formed from data sets are of two types: univariate and multivariate. A multivariate time series data set includes the number of individual vehicles, the average speed, maximum speed and minimum speed of these vehicles while univariate data set includes only average speed.

 While data preparation we create univariate time series data sets to predict the future average speed. If we use a multivariate time series data sets we create an input vector where four different variables oast values are considered together   to predict the future average speed.

Data sets are created with 6 past and 3 future values for multi-step forecasting with the window sliding method. Our goal is to estimate the average speed, for this purpose we want to predict the 3-hour future average speed for 3-hour using 6 hours of past data.



Train and Test Data Sets

5-month, 7-month and 12-month data sets were created by combining monthly data from each of the three individual sensors. Each sensor data set was used independently and was not correlated with each other. The data sets are split into 60% training data and 40% test data.

