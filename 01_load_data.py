import os
from datetime import datetime, timedelta
import pandas as pd
from matplotlib import pyplot as plt


# demo_name = "helsinki"
demo_name = "appliance"
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
    timestamp_col = "timestamp"

    df_sensors_dict = {}
    sensor_coords = {}

    for (dirpath, dirnames, filenames) in os.walk(data_folder):
        for fname in filenames:
            df_raw = pd.read_csv(os.path.join(data_folder, fname), delimiter=";")
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
            df_raw.set_index(keys='timestamp', inplace=True)
            sensor_board_name = fname.split("_")[0]
            sensor_board_num = '0' + sensor_board_name[1:]
            sensor_coords[(sensor_board_name[0], sensor_board_num[-2:])] = (float(fname.split("_")[1]),
                                                                            float(fname.split("_")[2]))
            df_sensors_dict.update({(sensor_board_name[0], sensor_board_num[-2:]): df_raw})

    df_multiindex = pd.concat(df_sensors_dict)
    df_multiindex.index = pd.MultiIndex.from_tuples(df_multiindex.index)
    df_multiindex.index.names = ['Cluster', 'Board', 'timestamp']

    df_multiindex.sort_index(inplace=True)

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


# print(f"DATASET NAME          : {demo_name} ")
# print(f"SENSOR BOARD NAMES ({len(df_sensors_dict)}): {list(df_sensors_dict.keys())} ")
# print(f"QUANTITIES         ({len(sensor_keys)}): {sensor_keys} ")

## PLOT
if demo_name=='helsinki':
    sensor_keys = ["O3", "PM10", "PM25"]
    n_sensor_keys = len(sensor_keys)
    cluster = 'V'
    boards = df_multiindex.loc[cluster].index.unique(level='Board')
    n_boards = len(boards)
    fig, axes = plt.subplots(n_boards, len(sensor_keys), sharex=True, figsize=(4*n_boards, 4*n_sensor_keys))
    for k, board in enumerate(boards):
        for l, sensor_key in enumerate(sensor_keys):
            axes[k][l].plot(
               df_multiindex.loc[cluster, board].index,
                df_multiindex.loc[cluster, board][sensor_key].values,
                label=sensor_key,
            )
            axes[k][l].grid()
            axes[k][l].set_ylabel(sensor_key)
            axes[k][l].set_title(f"Sensor Board ID: {cluster+board}")
else:
    n_sensor_boards = len(df_sensors_dict)
    fig, axes = plt.subplots(n_sensor_boards, 1, sharex=True, figsize=(23, 9))
    for ax, (sensor_board_name, df_sensor) in zip(axes, df_sensors_dict.items()):
        for sensor_key in sensor_keys:
            ax.plot(
                pd.to_datetime(df_sensor[timestamp_col]).values,
                df_sensor[sensor_key].values,
                label=sensor_key,
            )
        ax.grid()
        ax.set_ylabel("Values")
        ax.set_title(f"Sensor Board ID: {sensor_board_name}")

plt.show()
# axes[0].legend()
# axes[-1].set_xlabel("Timestamps")
# fig.tight_layout()
