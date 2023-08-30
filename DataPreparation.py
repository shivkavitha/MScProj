# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:46:18 2023

@author: ramas
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:48:23 2023

@author: ramas
"""

# for data operations and manipulations
import pandas as pd

# custom modules
from DataPreprocessing import DataPreprocessing
from DataFeatureEngineering import DataFeatureEngineering

# for handling data files
import os
from itertools import permutations

cwDataDir = os.getcwd() + '\Data\GPS_TrackingPETS2016'
subdirs = os.listdir(cwDataDir)
         
# To improve visualisation of dataframe
pd.set_option('display.max_columns',7)
            
class DataPreparation(object):        
    
    # Earth radius in meters
    Radius = 6372799.99374
    
    Safety_dist = 5 # in meters
    Floating_Speed = 1 # in meters
        
    # pre-defined behaviours
    semantics_dict = {'Increase_Speed' : 'Immoderate increase in speed', 'Decrease_Speed' : 'Immoderate decrease in speed', 'Abrupt_Bearing' : 'Abrupt change in direction', 'Immoderate_Bearing' : 'Immoderate change in direction', 'Approaching_Tween_Distance' : 'Approaching nearer boat', 'Seperating_Tween_Distance' : 'Seperating from nearer boat', 'Proximity' : 'Proximity'}      
    
    def __removeFile(self, filename):
        if os.path.exists(filename):
            os.remove(filename) 
                
    def __call__(self):
        
      self.__removeFile('Data/semantics/semantics.csv')     
     
      for subdir in subdirs:
         print('Current Sub-Directory - ' , subdir)
        
         for file_1, file_2 in permutations(os.listdir(cwDataDir + '\\' + subdir), r = 2):
             print('Current files in combination - ' , file_1, file_2)
          
             # Read only file in csv format
             if file_1.endswith(".csv") and file_2.endswith(".csv"):
                
                # Exclude behavioural observations of Partisan w.r.t other boats
                if not file_1.lower().endswith('partisan.csv'):
                   
                    # Exclude behavioural observations of BB and BP boats together
                     if not ((file_1.lower().endswith('bb.csv') & file_2.lower().endswith('bp.csv')) or (file_1.lower().endswith('bp.csv') & file_2.lower().endswith('bb.csv'))):
                                                   
                        # Load files into dataframes and add column headers
                        Obs_boat, boat_2 =  DataPreprocessing.loadFiles(subdir, file_1, file_2)
                       
                        # 1. Convert Timestamp to datetime dtype and drop rows with duplicate Timestamp keeping the last one
                        Obs_boat, boat_2 = DataPreprocessing.intToDatetime()
                       
                        # 2. Calculate distance, bearing and speed
                        parameters_df = DataFeatureEngineering.trajectoryParameters(Obs_boat, DataPreparation.Radius)
                       
                        # 3. Merge two dataframes and interpolate any NaN values
                        merged_df = DataFeatureEngineering.merge(Obs_boat, boat_2)
                        merged_df = DataFeatureEngineering.interpolate(merged_df)
                        
                        # Visualise trajectories
                        DataFeatureEngineering.visualiseTrajectories(file_1, file_2, merged_df)
                       
                        # 4. Calculate tween_distance
                        parameters_df = DataFeatureEngineering.mutualTrajectoryParameters(parameters_df, merged_df, DataPreparation.Radius)
                                                                
                        # 5. Observe changes in Tween_Distance, Speed and Bearing of non-floating boat
                        semantics_df = DataFeatureEngineering.semanticsParameters(parameters_df, DataPreparation.Floating_Speed, DataPreparation.Safety_dist)                        
                       
                        # Visualise anomalies in trajectory
                        if semantics_df is not None:
                            DataFeatureEngineering.visualiseChangePoints(merged_df, file_1 + ' and ' + file_2, semantics_df)
                       
                        # 6. Generate semantics describing the behaviour of the non-floating boat under observation
                        semantics = DataFeatureEngineering.generateSemantics(semantics_df, DataPreparation.semantics_dict)                                             
                       
                        # 7. Write semantics generated for a non-floating boat to a file
                        if semantics:                       
                            semantics_tofile = pd.DataFrame({'Files' : pd.Series(file_1 + ' and ' + file_2), 'Semantics' : pd.Series(semantics)})
                            semantics_tofile.to_csv('Data/semantics/semantics.csv', header = False, index = False, mode='a', sep=',')
               
            
                                   
if __name__ == '__main__':
    dataPreparation = DataPreparation()
    dataPreparation()





 