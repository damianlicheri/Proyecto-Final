import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pickle import load
import streamlit as st
from datetime import date
import joblib
from geopy.geocoders import Nominatim
from geosky import geo_plug
import requests
import json
import numpy as np

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://open-meteo.com/images/partly_cloudy.webp");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


list_cities_clean = []
list_cities_orig = geo_plug.all_Country_StateNames()
lista_diccionarios = json.loads(list_cities_orig)
all_keys = [key for dic in lista_diccionarios for key in dic.keys()]

for i in all_keys:
    var_index = int(all_keys.index(""+ i + ""))
    for x in lista_diccionarios[var_index].values():
        list_cities_ = list(x)
        list_cities_clean.extend(list(x))



list_cities_clean = [x for x in list_cities_clean if str(x) != 'N/A']

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
#latitud= st.number_input("Inserte la latitud")
#longitud= st.number_input("Inserte la longitud")
fecha=date.today().isoformat()

ciudad = st.selectbox(
    "Seleccione una ciudad",
    (list_cities_clean),
)
# Initialize Nominatim API
geolocator = Nominatim(user_agent="App_Tesis")
location = geolocator.geocode(ciudad)



# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": location.latitude,
	"longitude": location.longitude,
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl", "surface_pressure", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "vapour_pressure_deficit", "wind_gusts_10m", "soil_moisture_0_to_7cm"],
	"forecast_days": 3
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(3).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(5).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(7).ValuesAsNumpy()
hourly_vapour_pressure_deficit = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(9).ValuesAsNumpy()
hourly_soil_moisture_0_to_7cm = hourly.Variables(10).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm

hourly_dataframe = pd.DataFrame(data = hourly_data)

hourly_dataframe["hour"] = hourly_dataframe["date"].dt.hour
hourly_dataframe["period"] = hourly_dataframe["hour"].apply(lambda x: 1 if x <= 8 else (2 if x <= 16 else 3))
hourly_dataframe["date"]= hourly_dataframe["date"].dt.date
df_period = hourly_dataframe.groupby(['date', 'period']).agg({"temperature_2m":'mean',
'relative_humidity_2m':'mean',
'dew_point_2m':'mean',
'pressure_msl':'mean',
'surface_pressure':'mean',
'cloud_cover_low':'mean',
'cloud_cover_mid':'mean',
'cloud_cover_high':'mean',
'vapour_pressure_deficit':'mean',
'wind_gusts_10m':'mean',
'soil_moisture_0_to_7cm':'mean'})

df_datos= pd.DataFrame(df_period, columns=['temperature_2m', 'relative_humidity_2m','wind_gusts_10m'])

#Cargo modelo
model = joblib.load('model.pkl')



# creamos df y las columnas

fechas= df_period.index.get_level_values(0)
fechas = fechas.unique()
# Convertir la lista en un array de NumPy para facilitar el redimensionamiento



#Boton de prediccion

if st.button("Predict"):
    prediccion = model.predict(df_period)
    array_prediccion = np.array(prediccion)

# Redimensionar el array en una matriz con 3 columnas
    array_prediccion_reshaped = array_prediccion.reshape(-1, 3)

# Crear un DataFrame a partir de la matriz
    df = pd.DataFrame(array_prediccion_reshaped, columns=['Mañana', 'Tarde', 'Noche'], index=fechas)
    
    st.write("Predicción de lluvia para mañana(mm):", df)
    st.write("Datos adicionales de la predicción:", df_datos)





