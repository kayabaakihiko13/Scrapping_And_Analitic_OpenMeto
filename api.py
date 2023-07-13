import requests
import csv
from datetime import date
def open_meteo_api_to_csv(api_url, csv_file):
    response = requests.get(api_url)
    data = response.json()

    # Mengambil atribut header dari data JSON
    times = data['hourly']['time']

    # Membuat dictionary untuk atribut lainnya
    attributes = {}
    for attribute in data['hourly']:
        if attribute != 'time':
            attributes[attribute] = data['hourly'][attribute]

    # Membuka file CSV
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Menulis header CSV
        headers = ['Time'] + list(attributes.keys())
        writer.writerow(headers)

        # Menulis data ke dalam file CSV
        for i in range(len(times)):
            row = [times[i]]
            for attribute in attributes:
                row.append(attributes[attribute][i])
            writer.writerow(row)

    print("Konversi API JSON ke CSV selesai.")

def main():
    api_url = r"https://air-quality-api.open-meteo.com/v1/air-quality?latitude=52.52&longitude=13.41&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=Asia%2FBangkok&start_date=2023-01-01&end_date={}".format(date.today())
    csv_file = "data/air_quality.csv"
    open_meteo_api_to_csv(api_url, csv_file)

if __name__ == "__main__":
    main()
