import pathlib, json, os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import curve_fit
from lmfit import Model
from lmfit import Parameters
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd


class Initializer:
    '''
    Folder .json files are read and prepared for further analysis.

    output: pd.DataFrame
    '''
    def __init__(self, folderPath):
        self.folderPath = pathlib.Path(folderPath)
        self.ignore_files = []
        self.raw_data = {}

    def check_input_files(self):
        for file in self.folderPath.rglob('*.json'):
            if file.stat().st_size < 5000:
                self.ignore_files.append(file)

    def read_json(self, file):
        with open(file) as f:
            data = json.load(f)
        return data

    def prepare_rawdata(self):
        self.check_input_files()
        dfs = []
        for file in self.folderPath.rglob('*.json'):
            if file in self.ignore_files:
                continue
            data = self.read_json(file)

            df = pd.DataFrame.from_dict(data['raw_data'])
            df['datetime'] = pd.to_datetime(df['datetime_utc'])
            df['id'] = file.stem

            df['timedelta'] = df['datetime'] - df['datetime'].min()
            df['timedelta'] = df['timedelta'].apply(lambda x: x.seconds)

            config = data['config']

            df['alpha'] = config['alpha']
            df['c_s'] = config['c_s']
            df['c_c0'] = config['c_c0']
            df['deadband'] = config['deadband']
            try:
                df['disturbance_intensity'] = config['disturbance_intensity']
                df['disturbance_starting_point'] = config['disturbance_starting_point']
                df['add_noise'] = config['add_noise']
                df['curvature'] = config['curvature']
            except:
                pass
            dfs.append(df)
        df_data = pd.concat(dfs)
        df_data.sort_values(by='datetime', inplace=True)
        return df_data
    
    # def prepare_config_rawdata(self):
