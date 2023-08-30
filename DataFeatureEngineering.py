# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:47:45 2023

@author: ramas
"""

# for data operations and manipulations
import pandas as pd

# custom modules
import MaritimeUtils as util

# for mathematical operations
import numpy as np

# for change point detection
import ruptures as rpt

# for data visuals
import matplotlib.pyplot as plt

class DataFeatureEngineering(object):
    
    def merge(Obs_boat, boat_2):
        # Data of two boats merged
        return pd.DataFrame({'Obs_boat_Latitude' : Obs_boat['Obs_boat_Latitude'], 'Obs_boat_Longitude' : Obs_boat['Obs_boat_Longitude'], 'Boat2_Latitude' : boat_2['Boat2_Latitude'], 'Boat2_Longitude' : boat_2['Boat2_Longitude']})               
        
    def interpolate(merged_df):
        merged_df.interpolate(method= 'linear', limit_direction='both', axis = 0, inplace = True) # limit_direction='both' is required to interpolate first and last record
        print('\nmerging data of two boats:\n' , merged_df)
        
        return merged_df
    
    def visualiseTrajectories(file_1, file_2, merged_df):
        util.pl_trajectories(merged_df, file_1 + ' and ' + file_2)        
           
    def __calcTweenDistance(parameters_df, df, radius):     
        delta_Lat = np.radians(df['Obs_boat_Latitude'] - df['Boat2_Latitude'])
        delta_Lon = np.radians(df['Obs_boat_Longitude'] - df['Boat2_Longitude'])
        start_lat = np.radians(df['Boat2_Latitude'])
        end_lat = np.radians(df['Obs_boat_Latitude'])
     
        # Haversine formula
        a = np.sin(delta_Lat/2)**2 + np.cos(start_lat)*np.cos(end_lat)*np.sin(delta_Lon/2)**2
        c = 2*np.arcsin(np.sqrt(a))        
         
        parameters_df['Tween_Distance'] = radius * c # in miles
        print('\nTween distances\n' , parameters_df)
        
        return parameters_df
        
    def __calcDistance(parameters_df, df, radius):     
        delta_Lat = np.radians(df['Obs_boat_Latitude'].diff())
        delta_Lon = np.radians(df['Obs_boat_Longitude'].diff())
        start_lat = np.radians(df['Obs_boat_Latitude'].shift(1)) # Previous row
        end_lat = np.radians(df['Obs_boat_Latitude'])
                         
        # Haversine formula
        a = np.sin(delta_Lat/2)**2 + np.cos(start_lat)*np.cos(end_lat)*np.sin(delta_Lon/2)**2
        c = 2*np.arcsin(np.sqrt(a))        
        
        parameters_df['Distance'] = radius * c # in miles
        parameters_df['Distance'].fillna(0, inplace = True) # First row of the distance is always NaN which is replaced by 0.
        print('\nDistances\n' , parameters_df)
        
        return parameters_df
    
    def __calcSpeed(parameters_df, df):
        parameters_df['Timestamp'] = df.index
        time = parameters_df['Timestamp'].diff() / pd.Timedelta(seconds=1) # Convert to seconds
        parameters_df['Speed'] = parameters_df['Distance'] / time 
        parameters_df['Speed'].fillna(0, inplace = True) # First row of the speed is always NaN which is replaced by 0.
        parameters_df.drop(['Timestamp'], axis = 1, inplace = True)
        print('\nSpeeds\n' , parameters_df)
        
        return parameters_df
    
    def __calcBearing(parameters_df, df):        
        delta_Lon = np.radians(df['Obs_boat_Longitude'].diff())
        lat1 = np.radians(df['Obs_boat_Latitude'])
        lat2 = np.radians(df['Obs_boat_Latitude'].shift(1)) # Previous row
        
        x = np.sin(delta_Lon) * np.cos(lat2);
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_Lon);
        parameters_df['Bearing'] = np.rad2deg(np.arctan2(x, y))
        # df['Bearing'].fillna(0, inplace = True) # First row of the Bearing is always NaN which is replaced by 0.
        print('\nBearings\n' , parameters_df)
        
        return parameters_df
    
    def trajectoryParameters(df, radius):        
        # Create new dataframe for parameters being analysed
        parameters_df = pd.DataFrame(columns = ['Distance', 'Speed', 'Bearing', 'Tween_Distance'])         
        
        # a) Calculate Distance (between current location and previous location i.e., between current row and previous row)
        parameters_df = DataFeatureEngineering.__calcDistance(parameters_df, df, radius)
        
        # b) Calculate Speed
        parameters_df = DataFeatureEngineering.__calcSpeed(parameters_df, df)
        
        # c) Calculate Bearing between 0° to 180° with 0° being the North and –ve denoting for South and West sides.        
        parameters_df = DataFeatureEngineering.__calcBearing(parameters_df, df)
        
        return parameters_df
    
    def mutualTrajectoryParameters(parameters_df, merged_df, radius):        
        # d) Calculate Tween_Distance (distance between the two boats)
        parameters_df = DataFeatureEngineering.__calcTweenDistance(parameters_df, merged_df, radius) 
        
        return parameters_df
    
    def __changePointDetection(series, colName):
        df = series.to_frame(name = colName)
        
        algo = rpt.Pelt(model="l2").fit(df)
        
        T, d = df.shape  # number of samples, dimension
        std = df[colName].std()
        pen_bic = np.square(std) * np.log(T) * d # hyper parameter to optimise number of change points i.e., higher the 'pen_bic' value lower the number of change points detected.
        result = algo.predict(pen_bic)
        print('result', result)

        # display
        rpt.display(df, result)
        plt.show()
        
        zero_index_lst = [] # re-create list as zero index
        for id in result:
            zero_index_lst.append(id - 1)
       
        change_points = df.iloc[zero_index_lst]
        print('change_points for', colName, ':\n' , change_points)
        
        return change_points
    
    def semanticsParameters(parameters_df, floating_Speed, safety_dist):
        # Generate semantics only for non-floating boat
        if (parameters_df.Speed > floating_Speed).any():
            speed_df = DataFeatureEngineering.__changePointDetection(parameters_df.Speed, 'Speed')
            if ~ speed_df.empty:
                Increase_Speed = speed_df.loc[speed_df['Speed'] > speed_df['Speed'].shift()]['Speed']
                Decrease_Speed = speed_df.loc[speed_df['Speed'] < speed_df['Speed'].shift()]['Speed']
            
            bearing_df = DataFeatureEngineering.__changePointDetection(parameters_df.Bearing, 'Bearing')
            if ~ bearing_df.empty:
                Abrupt_Bearing = bearing_df.iloc[np.where(np.diff(np.sign(bearing_df['Bearing'])))[0] + 1]['Bearing']
                Immoderate_Bearing = bearing_df[~bearing_df.index.isin(Abrupt_Bearing.index)]['Bearing']
            
            tween_df = DataFeatureEngineering.__changePointDetection(parameters_df.Tween_Distance, 'Tween_Distance')
            if ~ tween_df.empty:
                Approaching_Tween_Distance = tween_df.loc[(tween_df['Tween_Distance'] < tween_df['Tween_Distance'].shift())]['Tween_Distance']              
                Seperating_Tween_Distance = tween_df.loc[(tween_df['Tween_Distance'] > tween_df['Tween_Distance'].shift())]['Tween_Distance']             
                Proximity = tween_df.loc[(tween_df['Tween_Distance'] <= safety_dist)]['Tween_Distance']               
                            
            # Capture observed changes
            semantics_df = pd.DataFrame({'Increase_Speed' : Increase_Speed, 'Decrease_Speed' : Decrease_Speed, 'Abrupt_Bearing' : Abrupt_Bearing, 'Immoderate_Bearing' : Immoderate_Bearing, 'Approaching_Tween_Distance' : Approaching_Tween_Distance, 'Seperating_Tween_Distance' : Seperating_Tween_Distance, 'Proximity' : Proximity})                  
            print('\nSemantics Parameters\n' , semantics_df)
            
            return semantics_df
        
    def generateSemantics(semantics_df, semantics_dict):  # semantics_dict provides pre-defined behaviours      
       
        # semantics describing the behaviour of the boat under observation
        semantics = ''
        
        # If the boat is floating then no semantics are generated
        if semantics_df is None:
            return semantics
            
        semantics_df.sort_index(inplace = True)
        
        # Check if 'approaching' appears in succession
        containsAppoach = False        
        approach_text = semantics_dict.get('Approaching_Tween_Distance')
        
        # Check if 'sperating' appears in succession
        containsSeperate = False        
        seperate_text = semantics_dict.get('Seperating_Tween_Distance')
        
        last_index = len(semantics_df) # last row index of the dataframe
        for index, row in semantics_df.iterrows():
             # semantics describing each row 
             semantics_row = ''
             
             # list of behaviours observed in each row
             text = []
             
             for col in semantics_df.columns:                
               if ~ np.isnan(row[col]):
                   text.append(semantics_dict.get(col))
             
             app_sep_text = '' # text for approaching or seperating
             
             # length of the list can't be zero since every row in semantics_df depicts some change in atleast one paramater or is floating
             if len(text) == 1:
                 semantics_row = text[0]
                 
             else:                 
                 if approach_text in text:
                     app_sep_text = approach_text
                     
                 elif seperate_text in text:
                     app_sep_text = seperate_text
                 
                 if bool(app_sep_text):
                     text.pop(text.index(app_sep_text))
                     text.insert(0, app_sep_text)                     
                                
                 last_item = text[-1] # last element of the list
                 for txt in text:
                     if txt == app_sep_text:
                        semantics_row =  semantics_row + txt + ' WITH '                        
                        
                     elif txt == last_item:                        
                        semantics_row =  semantics_row + txt # no 'AND' required
                       
                     else:
                        semantics_row =  semantics_row + txt + ' AND '           
            
             # If previous and current rows contain 'approaching...' then following row MUST say 'Keeps Approaching ...' for better readability
             if containsAppoach and approach_text.lower() in semantics_row.lower():
                semantics_row = 'Further ' + semantics_row[0].lower() + semantics_row[1:]
                
             # If previous and current rows contain 'Seperating...' then following row MUST say 'Further Seperating ...' for better readability
             if containsSeperate and seperate_text.lower() in semantics_row.lower():
                semantics_row = 'Further ' + semantics_row[0].lower() + semantics_row[1:]
                
             # Set to true for next row to know if previous row had a 'Approaching ...'
             if approach_text.lower() in semantics_row.lower():
                containsAppoach = True
               
             else:
                containsAppoach = False
               
             # Set to true for next row to know if previous row had a 'Seperating ...'
             if seperate_text.lower() in semantics_row.lower():
                 containsSeperate = True
               
             else:
                 containsSeperate = False
                 
             # Append semantics from all rows with a 'THEN'
             current_index = semantics_df.index.get_loc(index) + 1
             if current_index == last_index:
                semantics = semantics + semantics_row
                
             else:
                semantics = semantics + semantics_row + ' THEN '               
             
                
        print('\nSemantics Type 2 :\n', semantics)
        
        return semantics 
    
    
    def visualiseChangePoints(df, filenames, semantics_df):
                
        changedf = df[df.index.isin(semantics_df.index)]
        
        parameters = []
        for index, row in semantics_df.iterrows():
           text = list()
           for col in semantics_df.columns:                
              if ~ np.isnan(row[col]):
                text.append(col)
                
           parameters.append(text)
           
        changedf['parameters'] = parameters
        print('changedf with parameters', changedf)
                
        mapping = {'Increase_Speed' : 'darkred', 'Decrease_Speed' : 'darkgreen', 'Abrupt_Bearing' : 'darkorange', 'Immoderate_Bearing' : 'rosybrown', 'Approaching_Tween_Distance' : 'magenta', 'Seperating_Tween_Distance' : 'darkturquoise', 'Proximity' : 'black'}
        
        util.pl_changepoints(df, changedf, filenames, mapping)