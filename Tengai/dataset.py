import requests
import pandas as pd
from datetime import date
def AirQuality(lat:float,long:float):
    api_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={str(lat)}&longitude={str(long)}"
    api_url += f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,aerosol_optical_depth,dust,ammonia"
    api_url += f"&start_date=2023-01-01&end_date={date.today()}"
    # how to get respon api in form json file
    respon = requests.get(api_url)
    data = respon.json()
    memo:dict= {}
    for feature in data["hourly"]:
        memo[feature] = data["hourly"][feature]
    # change datatype memo from dict to Dataframe
    result = pd.DataFrame(memo)
    return result


def Weather(lat:float,long:float):
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={str(lat)}&longitude={str(long)}"
    api_url += f"&hourly=temperature_2m,precipitation_probability,precipitation,rain,pressure_msl,surface_pressure"
    api_url += f",cloudcover,windspeed_10m&start_date=2023-01-01&end_date={date.today()}"
    response = requests.get(api_url)
    data = response.json()
    # give a blank variabel to saving feature in data["hourly"]
    memo:dict = {}
    for feat in data["hourly"]:
        memo[feat] = data["hourly"][feat]
    result = pd.DataFrame(memo)
    return result

def ClimateChange(lat:float,long:float):
    api_url = f"https://climate-api.open-meteo.com/v1/climate?latitude={str(lat)}&longitude={str(long)}&start_date=1950-01-01&end_date=2023-10-04"
    api_url += f"&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily=temperature_2m_max"
    response = requests.get(api_url)
    data = response.json()
    # saving data with blank variabel
    memo = {}
    for feat in data["daily"]:
        memo[feat] = data["daily"][feat]
    # change from dict to DataFrame
    result = pd.DataFrame(memo)
    return result


if __name__ == "__main__":
    print(AirQuality(-8.2325,114.3576))
    print(Weather(-8.2325,114.3576))
    print(ClimateChange(-8.2325,114.3576))