# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:48:23 2023

@author: ramas
"""

# for data operations and manipulations
import pandas as pd

# for handling data files
import os

cwd = os.getcwd() + '\Data\GPS_TrackingPETS2016'
            
class DataPreprocessing(object): 
    
    # 'Obs_boat' refers to the boat under observation
    Obs_boat = pd.DataFrame(columns=['Obs_boat_Latitude','Obs_boat_Longitude', 'Timestamp'])
    # Second boat with which the behaviour of first boat is being observed
    boat_2 = pd.DataFrame(columns=['Boat2_Latitude','Boat2_Longitude', 'Timestamp']) 
            
    def loadFiles(dir, file_1, file_2):
        DataPreprocessing.Obs_boat = pd.read_csv(cwd + '\\' + dir + '\\' + file_1, names=['Obs_boat_Latitude','Obs_boat_Longitude', 'Timestamp'])
        DataPreprocessing.boat_2 = pd.read_csv(cwd + '\\' + dir + '\\' + file_2, names=['Boat2_Latitude','Boat2_Longitude', 'Timestamp'])           
        
        return DataPreprocessing.Obs_boat, DataPreprocessing.boat_2
    
    def intToDatetime():        
        # Convert Timestamp from 'int' to 'datetime' dtype by removing tailing 0's
        DataPreprocessing.Obs_boat['Timestamp'] = pd.to_datetime(pd.to_numeric(DataPreprocessing.Obs_boat['Timestamp'].astype(str).str[:-3]), unit='s')
        DataPreprocessing.boat_2['Timestamp'] = pd.to_datetime(pd.to_numeric(DataPreprocessing.boat_2['Timestamp'].astype(str).str[:-3]), unit='s')
        
        # Set Timestamp as index of the dataframe
        DataPreprocessing.Obs_boat.set_index('Timestamp', inplace=True)
        DataPreprocessing.boat_2.set_index('Timestamp', inplace=True)
        
        # Remove rows with duplicate Timestamps
        DataPreprocessing.Obs_boat = DataPreprocessing.Obs_boat[~DataPreprocessing.Obs_boat.index.duplicated(keep='last')]
        DataPreprocessing.boat_2 = DataPreprocessing.boat_2[~DataPreprocessing.boat_2.index.duplicated(keep='last')]

        # Sort the two dataframes by Timestamp
        DataPreprocessing.Obs_boat.sort_index(inplace = True)
        DataPreprocessing.boat_2.sort_index(inplace = True)
        
        return DataPreprocessing.Obs_boat, DataPreprocessing.boat_2
 