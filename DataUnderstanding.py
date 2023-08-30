# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:56:59 2023

@author: ramas
"""
# for handling data files
import os

# for data operations and manipulations
import pandas as pd

# custom modules
import MaritimeUtils as util

# for statistical tests
from statsmodels.tsa.stattools import adfuller

cwDataDir = os.getcwd() + '\Data\GPS_TrackingPETS2016'
subdirs = os.listdir(cwDataDir)

class DataUnderstanding(object):  
  
  def __checkMissingValues(self, df):      
      if df.isnull().sum().all():
         print('\nThere is Missing Value: \n')
         
  def __checkDuplicates(self, df):      
      if df.duplicated().all():
         print('\nThere is Duplicate: \n')
         
  def __checkDataTypes(self, df):     
      print('\ncolumns dtype: \n' , df.dtypes)  
     
  def __checkOutliers(self, df, file):
      # a) plot scatter matrix to visualise trajectories and outliers
      util.pl_scatterplot(df, file)
      
      # b) Visual Boxplot
      util.pl_boxplot(df, ['Longitude', 'Latitude', 'Timestamp'], file)
      
  def __checkNoise(self, df, file):
      # Check auto-correlation between lags
      util.pl_acf(df, file, ['Longitude', 'Latitude'])
      
      # Statistical test for stationarity
      result = adfuller(df.Latitude)
      print('ADF Statistic: %f' % result[0])
      print('p-value: %f' % result[1])
      print('Critical Values:')
      for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))           
      
      # Data Distributions
      util.pl_hist(df, ['Latitude','Longitude', 'Timestamp'], file)
           
  def exploreData(self):    
     for subdir in subdirs:
        print('Current Sub-Directory - ' , subdir)
        
        for file in os.listdir(cwDataDir + '\\' + subdir):
            
            df = pd.read_csv(cwDataDir + '\\' + subdir + '\\' + file, names=['Latitude','Longitude', 'Timestamp'])
            # 1. Checking for Missing Values            
            self.__checkMissingValues(df)
            
            # 2. Checking for Duplicates
            self.__checkDuplicates(df)
               
            # 3. Check columns dtype       
            self.__checkDataTypes(df)                   
            
            # 4. Checking for Outliers
            self.__checkOutliers(df, file)
            
            # 5. Checking for Noise 
            self.__checkNoise(df, file)
            
  
            
if __name__ == '__main__':
    DataUnderstanding().exploreData()