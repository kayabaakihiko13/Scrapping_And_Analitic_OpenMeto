import requests
import csv

def open_meteo_api_to_csv(api_url:str, csv_file:str):
    """
    this function for scrapping data weather in open-meteo.
    API data form JSON format we change to csv file
    Args:
        api_url (str): this parameter is include your
                       API KEY ACCESS
        csv_file (str): this parameter for what your file save
    """
    response = requests.get(api_url)
    data = response.json()

    # Mengambil data waktu, nilai pm10, dan nilai pm2_5
    times = data['hourly']['time']
    pm10_values = data['hourly']['pm10']
    pm2_5_values = data['hourly']['pm2_5']

    # Membuka file CSV
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Menulis header CSV
        headers = ['Time', 'PM10', 'PM2.5']
        writer.writerow(headers)

        # Menulis data ke dalam file CSV
        for i in range(len(times)):
            row = [times[i], pm10_values[i], pm2_5_values[i]]
            writer.writerow(row)
    # message for in Converst API JSON to CSV is done
    print("Convert API JSON To CSV is Compilated.")

# get data 
if __name__ =="__main__":
    open_meteo_api_to_csv('https://air-quality-api.open-meteo.com/v1/air-quality?latitude=-7.199997&longitude=112.80002&hourly=pm10,pm2_5&timezone=auto&start_date=2023-01-01&end_date=2023-07-13',
           'data/air_quality.csv')
