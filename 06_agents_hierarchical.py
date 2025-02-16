## Implement Aggregator Agent & online machine learning

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from river import anomaly


class SensorAgent(AgentMET4FOF):
    def init_parameters(self, df_historical=None, sensor_keys=["O3", "PM10", "PM25"]):
        self.df_historical = df_historical
        self.max_len = len(self.df_historical)
        self.current_index = 0

        if sensor_keys is None:
            self.sensor_keys = df_historical.columns
        else:
            self.sensor_keys = sensor_keys

    def agent_loop(self):
        if self.current_state == "Running":
            current_row = self.df_historical.iloc[self.current_index]
            self.current_index += 1

            output = {key: current_row[key] for key in self.sensor_keys}
            self.send_output(output)


class OCSVMAgent(AgentMET4FOF):

    def init_parameters(self, n_models=3):
        self.models = [
            anomaly.QuantileFilter(anomaly.OneClassSVM(nu=0.2), q=0.995)
            for i in range(n_models)
        ]

        self.first_fit = True

    def on_received_message(self, message):
        self.buffer.store(message["from"], message["data"])

        self.log_info(self.buffer.values())
        if self.buffer_filled(agent_name=message["from"]):
            self.log_info("BUFFER FULL")
            sensor_zscore = {}

            for _, sensor_buffer in self.buffer.items():
                for sensor_i, (sensor_type, sensor_reading) in enumerate(
                    sensor_buffer.items()
                ):
                    sensor_reading_ = np.array(sensor_reading)
                    sensor_reading_ = sensor_reading_[~np.isnan(sensor_reading)][
                        :, np.newaxis
                    ]  ## drop nan values

                    ## has valid values
                    if len(sensor_reading_) > 0:

                        ## first time fitting
                        if self.first_fit:
                            self.log_info("FITTING OCSVM!")
                            for data in sensor_reading_:
                                data_ = {"feature": data}
                                self.models[sensor_i].learn_one(data_)

                        ## incremental fit
                        else:
                            data_ = {"feature": sensor_reading_[-1]}

                        anomaly_score = self.models[sensor_i].score_one(data_)
                        anomaly_class = self.models[sensor_i].classify(anomaly_score)
                        sensor_zscore.update({sensor_type: anomaly_class})

            self.send_output(sensor_zscore)


class AggregatorAgent(AgentMET4FOF):
    ## Compute average of the values received
    def init_parameters(self, aggregate_keys=["O3", "PM10", "PM25"], max_seconds=2):
        self.aggregate_keys = aggregate_keys
        self.max_seconds = max_seconds

    def on_received_message(self, message):
        ## get number of agents that are connected to it
        n_input_agents = len(self.get_attr("Inputs"))

        ## store into buffer
        message["data"].update({"timestamp": datetime.now()})

        self.buffer.store(message["from"], message["data"])

        self.log_info(f"{self.buffer.values()}")

        ## compute mean of the latest value
        buffer_values = list(self.buffer.values())

        ## check 1: buffer len is the same
        if len(buffer_values) == n_input_agents:

            ## check 2: timestamp is now
            time_diffs = np.array(
                [data["timestamp"][-1].timestamp() for data in buffer_values]
            )
            time_diffs -= time_diffs[0]
            time_diffs = np.abs(time_diffs)

            ## all agent timestamps are synced
            if np.max(time_diffs) < self.max_seconds:
                output_dict = {}

                ## aggregate
                for key in self.aggregate_keys:
                    sensor_data = np.array([data[key][-1] for data in buffer_values])
                    self.log_info(f"COLLECTED DATA: {sensor_data}")
                    agg_data = np.nanmean(sensor_data)
                    output_dict.update({key: agg_data})

                ## send agg data
                self.send_output(output_dict)

            else:
                self.log_info(f"TIME STAMP SYNC ERROR: {np.max(time_diffs)}")


def main():
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
    window_size = 10

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
            df_sensor_board["datetime_round"] = df_sensor_board["datetime"].dt.ceil(
                "min"
            )
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
    print(
        f"SENSOR BOARD NAMES ({len(df_sensors_dict)}): {list(df_sensors_dict.keys())} "
    )
    print(f"QUANTITIES         ({len(sensor_keys)}): {sensor_keys} ")

    ## LEVEL1: total number at bottom level of hierarchy
    n_bottom = len(df_sensors_dict)

    ###=======================
    # start agent network server

    ## CREATE 3 LEVEL HIERARCHY
    ## LEVEL3-TOP, LEVEL2-MIDAGG, LEVEL1-BOTTOM

    agentNetwork = AgentNetwork(backend="mesa", dashboard_max_monitors=20)

    top_aggregator_agent = agentNetwork.add_agent(
        agentType=AggregatorAgent,
        buffer_size=5,
        name="TopLevelAggregator",
        aggregate_keys=sensor_keys,
    )
    top_agg_monitor_agent = agentNetwork.add_agent(
        name="TopMonitor", agentType=MonitorAgent
    )

    agentNetwork.bind_agents(top_aggregator_agent, top_agg_monitor_agent)

    ## CREATE LEVEL2: MID-AGG
    n_mid_agg = 2

    print(f"N BOTTOM: {df_sensors_dict}")

    mid_groups = np.array_split(np.arange(n_bottom), n_mid_agg)

    for mid_i, mid_group in enumerate(mid_groups):

        mid_aggregator_agent = agentNetwork.add_agent(
            name=f"MidLevelAggregator({mid_i+1})",
            agentType=AggregatorAgent,
            buffer_size=5,
            aggregate_keys=sensor_keys,
        )

        mid_agg_monitor_agent = agentNetwork.add_agent(
            name=f"MidMonitor({mid_i+1})", agentType=MonitorAgent
        )

        df_sensor_items = list(df_sensors_dict.items())

        mid_coalition_agents = [mid_aggregator_agent, mid_agg_monitor_agent]

        agentNetwork.bind_agents(mid_aggregator_agent, top_aggregator_agent)
        agentNetwork.bind_agents(mid_aggregator_agent, mid_agg_monitor_agent)

        for sensor_i in mid_group:
            (sensor_name, df_sensor) = df_sensor_items[sensor_i]
            sensor_agent = agentNetwork.add_agent(
                agentType=SensorAgent, df_historical=df_sensor, sensor_keys=sensor_keys
            )

            zscore_agent = agentNetwork.add_agent(
                agentType=OCSVMAgent, buffer_size=window_size
            )

            agentNetwork.bind_agents(sensor_agent, zscore_agent)
            agentNetwork.bind_agents(zscore_agent, mid_aggregator_agent)

            mid_coalition_agents.append(sensor_agent)
            mid_coalition_agents.append(zscore_agent)

        agentNetwork.add_coalition(f"MidLevel_{mid_i+1}", mid_coalition_agents)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
