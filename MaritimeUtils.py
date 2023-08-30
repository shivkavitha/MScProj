# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 22:56:15 2023

@author: ramas
"""

# for data visuals
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as mpatches

from statsmodels.graphics.tsaplots import plot_acf

def pl_scatterplot(df, filename):
    plt.clf()
    plot = sns.pairplot(df)
    plot.fig.suptitle(filename)
    plt.tight_layout()
    plt.show()
    
def pl_acf(df, filename, columns):
    plt.clf()
    for col in columns:
        plot_acf(df[col])
        plt.title(label = 'Auto Correlation for ' + col + ' in ' + filename)
        plt.tight_layout()
        plt.show()

def pl_trajectories(df, filenames):
    plt.clf()
    ax = plt.gca()
    df.plot(y = 'Obs_boat_Latitude' , x = 'Obs_boat_Longitude', ax = ax, label = filenames.split('and')[0].split('.')[0] + '\n(Under Observation)')
    df.plot(y = 'Boat2_Latitude' , x = 'Boat2_Longitude', ax = ax, label = filenames.split('and')[1].split('.')[0])
    plt.xlabel('Longitude')    
    plt.ylabel('Latitude')
    plt.legend()
    plt.xticks(rotation = 25)
    plt.yticks(rotation = 25)
    
    plt.title(filenames)
    plt.tight_layout()
    plt.show()   
    
def pl_changepoints(df, changedf, filenames, markerMapping):
    plt.clf()
    ax = plt.gca()    
    
    df.plot(y = 'Obs_boat_Latitude' , x = 'Obs_boat_Longitude', ax = ax, label = filenames.split('and')[0].split('.')[0] + ' (Under Observation)' + '\nw.r.t' + filenames.split('and')[1].split('.')[0])
    df.plot(y = 'Boat2_Latitude' , x = 'Boat2_Longitude', ax = ax, label = filenames.split('and')[1].split('.')[0])
    plt.xticks(rotation = 25)
    plt.yticks(rotation = 25)
    plt.xlabel('Longitude')    
    plt.ylabel('Latitude')
    plt.title('Change Points in Trajectory')   
    
    handles = []
    dupLabels = []
    
    for index, row in changedf.iterrows():
        
        markers = []
        for par in row['parameters']:         
           color = markerMapping.get(par)
           markers.append(color)
           if par not in dupLabels:
              handles.append(mpatches.Patch(color=color, label=par))
              dupLabels.append(par)      
      
        for c in markers:            
            ax.scatter(y = row['Obs_boat_Latitude'] , x = row['Obs_boat_Longitude'], c = c, marker='o', s=30, linestyle = 'None') # facecolor='white', edgecolor = 'black',
            temp = pickle.dumps(ax)
            ax = pickle.loads(temp)    
   
    plt.legend(handles=handles, framealpha=0.5, frameon=True)
    plt.show()
    
def pl_boxplot(df, columns, filename):
    plt.clf()
    for col in columns:
        df.boxplot(column = col)
        plt.yticks(rotation = 25)
        plt.title(label = filename)
        plt.tight_layout()
        plt.show()   
    
def pl_hist(df, columns, filename):
    plt.clf()
    for col in columns:
        plt.xticks(rotation = 25)
        plt.yticks(rotation = 25)
        
        plt.ylabel('Count')
        plt.xlabel(col)
           
        plt.title(label = filename)
        
        plt.hist(df[col])
        
        plt.tight_layout()
        plt.show()
        
        
def split_train_test(x, y, stratifyColumn):
    from sklearn.model_selection import train_test_split
    # retaining 33% test data across all models, hence hard-coded
    return train_test_split(x, y, test_size = 0.33, random_state = 0, stratify = stratifyColumn)