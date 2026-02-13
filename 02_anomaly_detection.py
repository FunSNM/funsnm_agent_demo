import os
from datetime import datetime, timedelta

import pandas as pd
from matplotlib import pyplot as plt


def rolling_zscore(price_series, window=10):
    rolling = pd.Series(price_series).rolling(window=window)
    return ((price_series - rolling.mean()) / rolling.std()).values


demo_name = "aws-iot"
# demo_name = "appliance"
# demo_name = "solar_power"
# demo_name = "water_level"
# demo_name = "helsinki"

dataset_folders = {
    "water_level": "catalonia-water-resource-daily-monitoring",
    "solar_power": "solar-power-generation-data",
    "aws-iot": "environmental-sensor-data-132k",
    "appliance": "appliances-energy-prediction",
    "helsinki": "helsinki",
}
data_folder = dataset_folders[demo_name]

df_sensors_dict = {}

##=====LOAD DATASETS===========
if demo_name == "helsinki":
    sensor_keys = ["O3", "PM10", "PM25"]
    timestamp_col = "timestamp"

    filenames = [
        "J1_60.166443_24.921221_3m_q10.csv",
        "J11_60.148267_24.919831_2m_q10.csv",
        "J7_60.156552_24.917241_2m_q10.csv",
        "J2_60.155524_24.915056_3m_q10.csv",
        "J5_60.158452_24.921397_3m_q10.csv",
    ]

    for i, filename in enumerate(filenames):
        df_raw = pd.read_csv(os.path.join(data_folder, filename), delimiter=";")
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        sensor_name = filename.split(".")[0]
        df_sensors_dict.update({sensor_name: df_raw})

elif demo_name == "aws-iot":
    sensor_keys = ["co", "humidity", "lpg"]
    timestamp_col = "datetime_round"
    df_raw = pd.read_csv(os.path.join(data_folder, "iot_telemetry_data.csv"))

    # convert unix time to time of day
    start = datetime(1970, 1, 1)  # Unix epoch start time
    df_raw["datetime"] = df_raw.ts.apply(lambda x: start + timedelta(seconds=x))
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])

    df_sensors = df_raw.groupby("device")

    # Dictionary to store each device's DataFrame
    for device, df_sensor_board in df_sensors:
        df_sensor_board["datetime_round"] = df_sensor_board["datetime"].dt.ceil("min")
        df_sensor_board.drop(["device", "datetime", "ts"], inplace=True, axis=1)
        df_sensor_board = (
            df_sensor_board.groupby("datetime_round").mean().reset_index(drop=False)
        )
        df_sensors_dict.update({device: df_sensor_board})

elif demo_name == "appliance":
    sensor_keys = ["Temperature", "Humidity"]
    filename = "KAG_energydata_complete.csv"
    timestamp_col = "date"
    df_raw = pd.read_csv(os.path.join(data_folder, filename))

    sensor_cols = [["T1", "RH_1"], ["T2", "RH_2"], ["T3", "RH_3"]]
    sensor_names = ["kitchen_area", "living_room", "laundry_room"]
    for sensor_col, sensor_name in zip(sensor_cols, sensor_names):
        df_sensor_board = df_raw[[timestamp_col] + sensor_col]
        df_sensor_board.columns = [timestamp_col, "Temperature", "Humidity"]
        df_sensors_dict.update({sensor_name: df_sensor_board})

elif demo_name == "solar_power":
    sensor_keys = ["DC_POWER", "AC_POWER"]
    timestamp_col = "DATE_TIME"
    df_raw = pd.read_csv(os.path.join(data_folder, "Plant_1_Generation_Data.csv"))

    # convert unix time to time of day
    start = datetime(1970, 1, 1)  # Unix epoch start time
    df_raw["DATE_TIME"] = pd.to_datetime(df_raw["DATE_TIME"])

    df_sensors = df_raw.groupby("SOURCE_KEY")

    # Dictionary to store each device's DataFrame
    max_sensors = 10

    for sensor_i, (device, df_sensor_board) in enumerate(df_sensors):
        if sensor_i > max_sensors:
            continue
        df_sensor_board.drop(
            ["PLANT_ID", "TOTAL_YIELD", "DAILY_YIELD"], inplace=True, axis=1
        )
        df_sensor_board = df_sensor_board.sort_values("DATE_TIME").reset_index(
            drop=False
        )
        df_sensors_dict.update({device: df_sensor_board})

elif demo_name == "water_level":
    sensor_keys = ["reservoir_volume"]
    timestamp_col = "timestamp"
    filename = os.path.join(data_folder, "reservoir_sensors_reads.csv")
    sensor_names = ["reservoir"]

    df_raw = pd.read_csv(filename)
    df_raw.columns = ["timestamp"] + [col for col in df_raw.columns[1:]]
    sensor_cols = list(df_raw.columns[1:6])

    for sensor_name in sensor_cols:
        df_sensor_board = pd.DataFrame(df_raw[[timestamp_col, sensor_name]])
        df_sensor_board.columns = [timestamp_col, "reservoir_volume"]
        df_sensor_board = df_sensor_board.sort_values(timestamp_col).reset_index(
            drop=True
        )
        df_sensors_dict.update({sensor_name: df_sensor_board})

print(f"DATASET NAME          : {demo_name} ")
print(f"SENSOR BOARD NAMES ({len(df_sensors_dict)}): {list(df_sensors_dict.keys())} ")
print(f"QUANTITIES         ({len(sensor_keys)}): {sensor_keys} ")


##===APPLY ROLLING ZSCORE===
window = 10

n_sensor_boards = len(df_sensors_dict)

zscore_dicts = {}
timestamp_dicts = {}
for sensor_board_name, df_sensor_board in df_sensors_dict.items():
    zscore_dict = {}
    timestamp_dict = {}
    for quantity_col in sensor_keys:
        zscore = rolling_zscore(df_sensor_board[quantity_col].values, window=window)
        zscore_dict.update({quantity_col: zscore})
        timestamp_dict.update(
            {"timestamp": pd.to_datetime(df_sensor_board[timestamp_col]).to_numpy()}
        )
    zscore_dicts.update({sensor_board_name: zscore_dict})
    timestamp_dicts.update({sensor_board_name: timestamp_dict})

## PLOT
fig, axes = plt.subplots(n_sensor_boards, 1, sharex=True, figsize=(20, 9))
for ax, (sensor_board_name, zscore_dict) in zip(axes, zscore_dicts.items()):
    for quantity_col, zscore in zscore_dict.items():

        ax.plot(
            timestamp_dicts[sensor_board_name]["timestamp"], zscore, label=quantity_col
        )
    ax.grid()
    ax.set_ylabel("Z-Score")
    ax.set_title(f"Sensor Board ID: {sensor_board_name}")

axes[0].legend()
axes[-1].set_xlabel("Timestamps")
fig.tight_layout()
