import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pickle import load
import streamlit as st
from datetime import date
import joblib




# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
latitud= st.number_input("Inserte la latitud")
longitud= st.number_input("Inserte la longitud")
fecha=date.today().isoformat()


# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": latitud,
	"longitude": longitud,
	"hourly": ["temperature_2m","relative_humidity_2m", "dew_point_2m", "rain", "snowfall", "pressure_msl", "surface_pressure", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "vapour_pressure_deficit", "wind_gusts_10m", "soil_moisture_0_to_7cm"],
	"start_date": fecha,
	"end_date": fecha
}
responses = openmeteo.weather_api(url, params=params)
response = responses[0]


# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(4).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(7).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(8).ValuesAsNumpy()
hourly_vapour_pressure_deficit = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(10).ValuesAsNumpy()
hourly_soil_moisture_0_to_1cm = hourly.Variables(11).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_1cm

hourly_dataframe = pd.DataFrame(data = hourly_data)

hourly_dataframe["hour"] = hourly_dataframe["date"].dt.hour
hourly_dataframe["period"] = hourly_dataframe["hour"].apply(lambda x: 1 if x <= 8 else (2 if x <= 16 else 3))
hourly_dataframe["date"]= hourly_dataframe["date"].dt.date
df_period = hourly_dataframe.groupby(['date', 'period']).agg({"temperature_2m":'mean',
'relative_humidity_2m':'mean',
'dew_point_2m':'mean',
'rain':'mean',
'snowfall':'mean',
'pressure_msl':'mean',
'surface_pressure':'mean',
'cloud_cover_low':'mean',
'cloud_cover_mid':'mean',
'cloud_cover_high':'mean',
'vapour_pressure_deficit':'mean',
'wind_gusts_10m':'mean',
'soil_moisture_0_to_7cm':'mean'})


#Cargo modelo
model = joblib.load("model.pkl", "rb")



# creamos df y las columnas

df = pd.DataFrame()
df['Mañana'] = None
df['Tarde'] = None
df['Noche'] = None

#Boton de prediccion

if st.button("Predict"):
    prediccion = model.predict(df_period)
    #agrego a df la predicción
    df.loc['Prediccion'] = prediccion
    
    st.write("Predicción de lluvia para mañana:", df)


