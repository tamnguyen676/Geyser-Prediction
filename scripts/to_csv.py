from pytz import timezone
import pandas as pd
import datetime
import numpy as np

def is_pos_number(str):
    return str.replace('.', '', 1).isdigit()

def process_weather_csv():
    df = pd.read_csv('../csv/weather.csv', parse_dates=['datetime'])
    df['Wind Speed'] = df['Wind Speed'].replace('M', np.nan)
    df['Wind Speed'] = df['Wind Speed'].replace('m', np.nan)
    df['Wind Speed'] = df['Wind Speed'].fillna(method='ffill')
    df['Precip'] = df['Precip'].replace('M', np.nan)
    df['Precip'] = df['Precip'].replace('m', np.nan)
    df['Precip'] = df['Precip'].fillna(method='ffill')

    datetimes, rain_cumsums, wind_cumsums = [], [], []
    prev = None
    rain_cumsum = wind_cumsum = 0
    for idx, row in df.iterrows():
        date = row['Date']
        time = row['Time']
        date_time = datetime.datetime.strptime(f'{date} {time}', '%Y-%m-%d %H:%M')
        date_time = date_time.replace(tzinfo=timezone('UTC'))
        datetimes.append(date_time)
        date_time = row['datetime']
        if prev is None or date_time.day != prev.day or date_time.month != prev.month or date_time.year != prev.year:
            rain_cumsum = wind_cumsum = 0
        prev = date_time

        rain_cumsum += float(row['Precip'])
        wind_cumsum += float(row['Wind Speed'])
        rain_cumsums.append(rain_cumsum)
        wind_cumsums.append(wind_cumsum)

    df['datetime'] = datetimes
    df['Daily Precip Cumsum'] = rain_cumsums
    df['Daily Wind Speed Cumsum'] = wind_cumsums
    df = df.drop(columns=['Date', 'Time', 'Unnamed: 9'])
    df.to_csv('../weather.csv')

def merge_geyser_and_weather():
    geyser_df = pd.read_csv('../csv/old_faithful_nps.csv', parse_dates=['datetime'])
    weather_df = pd.read_csv('../csv/weather.csv', parse_dates=['datetime'])
    old_faithful_weather = pd.merge_asof(geyser_df, weather_df, on='datetime')
    old_faithful_weather = old_faithful_weather.rename(columns={'Unnamed: 0': 'Orig. Weather Idx'})
    old_faithful_weather.to_csv('../old_faithful_weather.csv')

merge_geyser_and_weather()
# process_weather_csv()