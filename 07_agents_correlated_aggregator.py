## Implement Correlated Aggregator Agent

import os
from datetime import datetime

import numpy as np
import pandas as pd
from agentMET4FOF.agents import MetrologicalAgent, MonitorAgent
from agentMET4FOF.network import AgentNetwork
from math import sin, cos, sqrt, atan2, radians
import re
import plotly.graph_objs as go
from scipy.linalg import solve


def custom_plot_function(data, sender_agent, cov_factor=2, sensor_keys=["O3", "PM10", "PM25"]):
    # Plot uncertainty weighted mean and confidence intervals for all measurands
    trace = []
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44),
              (214, 39, 40), (148, 103, 189), (140, 86, 75),
              (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
    for i, measurand in enumerate(sensor_keys):
        # Retrieve data
        mean = np.array(data[measurand + " - Mean"])
        uncertainty = np.array(data[measurand + " - Uncertainty"])
        time = np.array(data[measurand + " - Time"])

        # Determine boundaries of confidence interval
        lower = mean - cov_factor * uncertainty
        upper = mean + cov_factor * uncertainty

        # Define color codes
        color_tuple = colors[np.mod(i, len(colors) - 1)]
        color = "rgb" + str(color_tuple)
        color_rgba = "rgba" + str(color_tuple + (0.5,))

        # Define the traces
        trace.extend(
            [go.Scatter(
                x=time,
                y=mean,
                mode="lines",
                name=measurand,
                line=dict(color=color),
            ),
            go.Scatter(
                x=time,
                y=lower,
                mode="lines",
                showlegend=False,
                line=dict(color=color, dash='dash'),
            ),
            go.Scatter(
                x=time,
                y=upper,
                mode="lines",
                showlegend=False,
                line=dict(color=color, dash='dash'),
                fill='tonexty',  # Fill to trace above (upper)
                fillcolor=color_rgba  # color with 50% transparency
            )]
        )
    return trace


class SensorAgent(MetrologicalAgent):
    def init_parameters(self, df_historical=None, sensor_keys=["O3", "PM10", "PM25"],
                        metadata=None):
        super(SensorAgent, self).init_parameters(self)
        if metadata is None:
            metadata = {}
        self.df_historical = df_historical
        self.max_len = len(self.df_historical)
        self.current_index = 0
        self.metadata = metadata

        if sensor_keys is None:
            self.sensor_keys = df_historical.columns
        else:
            self.sensor_keys = sensor_keys

    def agent_loop(self):
        if self.current_state == "Running":
            current_row = self.df_historical.iloc[self.current_index]
            self.current_index += 1

            output = {key: current_row[key] for key in self.sensor_keys}
            self.send_output([output, self.metadata])


class BufferAgent(MetrologicalAgent):
    def on_received_message(self, message):
        self.buffer.store(message["from"], message["data"])

        self.log_info(self.buffer.values())
        if self.buffer_filled(agent_name=message["from"]):
            self.log_info("BUFFER FULL")
            sensor_output = {}
            for _, sensor_buffer in self.buffer.items():
                for sensor_type, sensor_reading in sensor_buffer.items():
                    sensor_output[sensor_type] = np.array(sensor_reading)
            self.send_output([sensor_output, message["metadata"]])


class CorrelatedAggregatorAgent(MetrologicalAgent):
    def init_parameters(self, aggregate_keys=["O3", "PM10", "PM25"], max_seconds=2, metadata={}):
        super(CorrelatedAggregatorAgent, self).init_parameters(self)
        self.aggregate_keys = aggregate_keys
        self.max_seconds = max_seconds
        self.metadata = metadata
        self.timestamps = {}
        self.plot_time = None

    def on_received_message(self, message):
        ## get number of agents that are connected to it
        n_input_agents = len(self.get_attr("Inputs"))

        ## store into buffer
        self.timestamps[message["from"]] = datetime.now().timestamp()

        self.buffer.store(message["from"], message["data"])
        if message["from"] not in self.metadata.keys():
            self.metadata[message["from"]] = message["metadata"]
        self.log_info(f"{self.buffer.values()}")

        # Define start time for plot if it is not defined yet
        # First plot point is start time of data + time dif between measurements * buffer size
        if self.plot_time is None:
            self.plot_time = self.metadata[message["from"]]["start_time"]
            self.plot_time += self.metadata[message["from"]]["time_dif"]*self.buffer_size

        ## compute mean of the latest value
        buffer_values = list(self.buffer.values())

        ## check 1: buffer len is the same
        if len(buffer_values) == n_input_agents:
            ## check 2: timestamp is now
            time_diffs = np.array(
                [timestamp for timestamp in self.timestamps.values()]
            )
            time_diffs -= time_diffs[0]
            time_diffs = np.abs(time_diffs)

            ## all agent timestamps are synced
            if np.max(time_diffs) < self.max_seconds:
                output_dict = {}

                # Check if all data sources have provided the same amount of datapoints
                data_length = [len(buffer_values[i][key])
                               for i in range(len(buffer_values))
                               for key in buffer_values[i].keys()]

                if any(np.diff(data_length) != 0):
                    return

                ## aggregate
                for key in self.aggregate_keys:
                    # Get sensor data and names
                    sensor_names = list(self.buffer.keys())
                    sensor_data = [self.buffer[sensor_name][key] for sensor_name in sensor_names]

                    # Remove nans from data and remove sensors that only have nan data
                    sensor_data_no_nan = [sensor_data_i[~np.isnan(sensor_data_i)] for sensor_data_i in sensor_data]
                    non_empty_idx = [len(sensor_data_i) > 0 for sensor_data_i in sensor_data_no_nan]
                    sensor_data_no_nan = [sensor_data_no_nan[i] for i in range(len(sensor_data_no_nan)) if non_empty_idx[i]]
                    sensor_names = [sensor_names[i] for i in range(len(sensor_names)) if non_empty_idx[i]]
                    self.log_info(f"COLLECTED DATA: {sensor_data_no_nan}")

                    # Calculate covariance matrix of the sensor data
                    cov_mat = self.get_sensor_covariance(sensor_data_no_nan, sensor_names)

                    # Calculate uncertainty weighted mean and its uncertainty
                    sensor_data_flat = np.concatenate(sensor_data_no_nan)
                    mean, mean_uncertainty = self.weighted_mean_cov(sensor_data_flat, cov_mat)
                    output_dict.update({key + " - Mean": mean,
                                        key + " - Uncertainty": mean_uncertainty,
                                        key + " - Time": self.plot_time})

                ## send agg data
                self.send_output((output_dict, None))

                # Update variables for next loop
                self.plot_time += self.metadata[message["from"]]["time_dif"]
                self.timestamps = {}
            else:
                self.log_info(f"TIME STAMP SYNC ERROR: {np.max(time_diffs)}")

    def get_sensor_distance(self, coords_1, coords_2):
        R = 6373.0  # km
        # Convert coordinates to radians
        lat1 = radians(coords_1[0])
        lon1 = radians(coords_1[1])
        lat2 = radians(coords_2[0])
        lon2 = radians(coords_2[1])

        # Determine distance between coordinates
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def get_distance_mat(self, sensor_data, sensor_names):
        # Set up empty matrix that will be filled
        nr_data_points = sum([len(sensor_data_i) for sensor_data_i in sensor_data])
        dist_mat = np.full((nr_data_points, nr_data_points), np.nan)
        nr_sensors = len(sensor_names)
        idx_end_1 = 0
        # Loop through the different sensors and add distance with respect to all sensors to distance matrix
        for i in range(nr_sensors):
            idx_start_1 = idx_end_1
            idx_end_2 = idx_end_1
            idx_end_1 += len(sensor_data[i])
            for j in range(i, nr_sensors):
                idx_start_2 = idx_end_2
                idx_end_2 += len(sensor_data[j])
                dist = self.get_sensor_distance(self.metadata[sensor_names[i]]["loc"],
                                                self.metadata[sensor_names[j]]["loc"])
                dist_mat[idx_start_2:idx_end_2, idx_start_1:idx_end_1] = dist
                dist_mat[idx_start_1:idx_end_1, idx_start_2:idx_end_2] = dist
        return dist_mat

    def get_time_dif_mat(self, sensor_data, sensor_names):
        # Set up empty matrix that will be filled
        nr_data_points = sum([len(sensor_data_i) for sensor_data_i in sensor_data])
        time_data = np.full(nr_data_points, np.nan)

        # Loop through the different sensors and add time difference with respect to all observations to time matrix
        idx = 0
        for i, sensor_name in enumerate(sensor_names):
            idx_new = idx + len(sensor_data[i])
            time_per_meas = self.metadata[sensor_name]["time_dif"].astype('timedelta64[m]') / 60  # time dif in hours
            time_data[idx:idx_new] = np.arange(len(sensor_data[i])) * time_per_meas
            idx = idx_new
        time_dif_mat = np.abs(time_data[:, None] - time_data[None, :])
        return time_dif_mat

    def kernel(self, dist_mat, time_dif_mat):
        # Kernel from paper: G. Kok, et al., "Modelling and determining correlations in sensor networks"
        # doi: https://doi.org/10.1016/j.measen.2024.101793
        # Independent part of the kernel
        dirac_time = time_dif_mat == 0
        dirac_space = dist_mat == 0
        independent_stdev = 0.63
        independent_cov = independent_stdev ** 2 * dirac_time * dirac_space

        # Decaying periodicity
        periodicity_stdev = 2.5
        length_scale_periodicity = 0.73
        length_scale_periodicity_decay = 320
        periodicity = 24  # Daily periodicity
        decaying_periodic_kernel = periodicity_stdev ** 2 * np.exp(
            -time_dif_mat ** 2 / (2 * length_scale_periodicity_decay ** 2) -
            2 * np.sin(np.pi * time_dif_mat / periodicity) ** 2 / length_scale_periodicity ** 2)
        # Medium term irregularities
        medium_irr_stdev = 3.0
        length_scale_medium_irr = 1.7
        mixture_par_medium_irr = 1.3
        medium_irr_kernel = (medium_irr_stdev ** 2) * (
                    1 + time_dif_mat ** 2 / (2 * length_scale_medium_irr ** 2 * mixture_par_medium_irr)) ** (
                                -mixture_par_medium_irr)

        # Time part of the kernel is some of decaying periodicity and medium term irregularities
        time_kernel = decaying_periodic_kernel + medium_irr_kernel

        # Space part of the kernel is a decaying kernel
        length_scale_dist = 1
        dist_kernel = np.exp(-dist_mat ** 2 / (2 * length_scale_dist ** 2))

        # Total kernel is space and time kernel multiplied summed with independent part of the kernel
        cov_mat = time_kernel * dist_kernel + independent_cov
        return cov_mat

    def get_sensor_covariance(self, sensor_data, sensor_names):
        # Get distance and time differences for all data points
        dist_mat = self.get_distance_mat(sensor_data, sensor_names)
        time_dif_mat = self.get_time_dif_mat(sensor_data, sensor_names)

        # Determine covariance based on distance and time differences
        cov_mat = self.kernel(dist_mat, time_dif_mat)
        return cov_mat

    def weighted_mean_cov(self, x, cov):
        # Instantiate objects
        x = np.asarray(x)
        cov = np.asarray(cov)
        ones = np.ones_like(x)

        # Solve C⁻¹x and C⁻¹1 without explicitly inverting C
        c_inv_x = solve(cov, x, assume_a='pos')
        c_inv_1 = solve(cov, ones, assume_a='pos')

        # Intermediate calculations
        numerator = np.dot(ones, c_inv_x)
        denominator = np.dot(ones, c_inv_1)

        # Determine uncertainty weighted mean and its standard deviation
        mean = numerator / denominator
        std = np.sqrt(1 / denominator)
        return mean, std


def main():

    demo_name = "helsinki"

    dataset_folders = {
        "helsinki": "helsinki",
    }
    data_folder = dataset_folders[demo_name]

    df_sensors_dict = {}
    window_size = 24

    ##=====LOAD DATASETS===========
    if demo_name == "helsinki":
        sensor_keys = ["O3", "PM10", "PM25"]
        timestamp_col = "timestamp"
        metadata = {}

        filenames = [
            "J1_60.166443_24.921221_3m_q10.csv",
            "J11_60.148267_24.919831_2m_q10.csv",
            "J7_60.156552_24.917241_2m_q10.csv",
            "J2_60.155524_24.915056_3m_q10.csv",
            "J5_60.158452_24.921397_3m_q10.csv",
        ]

        coord_search_string = r"(?P<lon>\d+\.\d+)_(?P<lat>\d+\.\d+)"
        for i, filename in enumerate(filenames):
            # Get data
            df_raw = pd.read_csv(os.path.join(data_folder, filename), delimiter=";")
            df_raw[timestamp_col] = pd.to_datetime(df_raw[timestamp_col])

            # Store data in dict
            sensor_name = filename.split(".")[0]
            df_sensors_dict.update({sensor_name: df_raw})

            # Get start timestamp
            start_time = df_raw[timestamp_col].iloc[0]

            # Find median time difference between observations (used to in kernel calculations)
            time_dif = np.array(df_raw[timestamp_col].iloc[1:]) - np.array(df_raw[timestamp_col].iloc[:-1])
            median_time_dif = np.median(time_dif.astype('timedelta64[m]'))

            # Find coordinates of sensor location (in filename)
            found_coords = re.search(coord_search_string, filename)
            if found_coords is None:
                raise Exception("Sensor coordinates must be provided in filenames")
            coords = (float(found_coords.group("lon")),
                      float(found_coords.group("lat")))

            # Save coordinates and time difference in metadata
            metadata[sensor_name] = {"loc": coords,
                                     "time_dif": median_time_dif,
                                     "start_time": start_time}
        if len(list(set([metadata[key]["start_time"] for key in metadata.keys()]))) != 1:
            raise Exception("Sensor start times must be equal")
    print(f"DATASET NAME          : {demo_name} ")
    print(
        f"SENSOR BOARD NAMES ({len(df_sensors_dict)}): {list(df_sensors_dict.keys())} "
    )
    print(f"QUANTITIES         ({len(sensor_keys)}): {sensor_keys} ")

    ###=======================
    # start agent network server
    agentNetwork = AgentNetwork(ip_addr='127.0.0.1', backend='MESA', dashboard_max_monitors=20)

    aggregator_agent = agentNetwork.add_agent(
        agentType=CorrelatedAggregatorAgent, aggregate_keys=sensor_keys, buffer_size=window_size
    )
    aggregator_agent.init_parameters(metadata=metadata)

    for sensor_i, (sensor_name, df_sensor) in enumerate(df_sensors_dict.items()):
        metadata_i = metadata[sensor_name]
        sensor_agent = agentNetwork.add_agent(
            agentType=SensorAgent, df_historical=df_sensor, sensor_keys=sensor_keys,
            metadata = metadata_i
        )

        buffer_agent = agentNetwork.add_agent(
            agentType=BufferAgent, buffer_size=window_size
        )

        monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent, name="Data sensor {:d}".format(sensor_i + 1))

        agentNetwork.bind_agents(sensor_agent, buffer_agent)
        agentNetwork.bind_agents(buffer_agent, monitor_agent)

        agentNetwork.add_coalition(
            f"SensorGroup_{sensor_i}", [sensor_agent, buffer_agent, monitor_agent]
        )

        agentNetwork.bind_agents(buffer_agent, aggregator_agent)

    agg_monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent,
                                               name="Aggregated mean (last {:d} datapoints)".format(window_size))
    agg_monitor_agent.init_parameters(custom_plot_function=custom_plot_function, cov_factor=2)
    agentNetwork.bind_agents(aggregator_agent, agg_monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
