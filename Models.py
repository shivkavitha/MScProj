# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:14:43 2023

@author: ramas
"""

# for mathematical operations
import numpy as np

# for reproducibility
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(89)
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

# for data operations and manipulations
import pandas as pd

# custom modules
import MaritimeUtils as util

# For deep learning models
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
import keras_tuner
import tensorflow_addons as tfa

# For plotting models
from keras.utils import plot_model
  
########## Train and Evaluate Deep learning Models #########   
class Models(object):
    
    x_train = pd.DataFrame()
    x_test  = pd.DataFrame()
    y_train  = pd.DataFrame()
    y_test  = pd.DataFrame()
    
    def __init__(self):
        
        # Read labeled semantics file                       
        semantics_df = pd.read_csv(os.getcwd() + '\\' + 'Data' + '\\' + 'Model' + '\\' + 'semantics.csv', names=['Semantics','Threat?'])
        print(semantics_df)        
        
        # Split dataset into features (X) and target (y)
        x = semantics_df.iloc[:, 0]
        y = semantics_df.iloc[:, 1]
        
        # convert labels into binary 0/1
        y = y.map(dict(Yes=1, No=0))        
        
        # Split X and Y into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = util.split_train_test(x, y, stratifyColumn = y)        
        
        # Verify split number of records for each target type        
        print('training\n', self.y_train.value_counts())
        print('testing\n', self.y_test.value_counts()) 
    
    def __tokenise(self, x_train, x_test):
        # Tokenize training dataset
        tokenizer = Tokenizer(num_words=500) # input data may contain possibly less than 500 words
        tokenizer.fit_on_texts(x_train)
        x_train_sequences = tokenizer.texts_to_sequences(x_train)
        max_train_length = max([len(s) for s in x_train_sequences])
        x_train_padded_data = pad_sequences(x_train_sequences, maxlen=max_train_length, padding="post") # padding="post" indicates pad zeros in the end
        
        # Tokenize testing dataset
        tokenizer.fit_on_texts(x_test)
        x_test_sequences = tokenizer.texts_to_sequences(x_test)
        max_test_length = max([len(s) for s in x_test_sequences])        
        x_test_padded_data = pad_sequences(x_test_sequences, maxlen=max_test_length, padding="post") # padding="post" indicates pad zeros in the end
        
        return x_train_padded_data, x_test_padded_data
    
    def __lstmModel(self):
        model = Sequential()
        model.add(Embedding(input_dim=500, output_dim=128, mask_zero=True)) # "mask_zero=True" tell the model that data is masked or padded
        model.add(LSTM(80))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, average=None, threshold = 0.5)])
        print(model.summary())
        
        # generate model diagram
        plot_model(model, to_file='lstm_plot.png', show_shapes=True, show_layer_names=True)
        
        return model
    
    ######### LSTM with Hold-Out approach #########
    def lstm_hold_out(self):        
        x_train_padded_data, x_test_padded_data = self.__tokenise(self.x_train, self.x_test)
        
        model = self.__lstmModel()        
        model.fit(x_train_padded_data, self.y_train, validation_data=(x_test_padded_data, self.y_test), epochs=15, batch_size=10)
        
        # Evaluate the model
        result = model.evaluate(x_test_padded_data, self.y_test, verbose=0)
        
        print("Loss: %.2f%%" % (result[0] * 100))
        print("Accuracy: %.2f%%" % (result[1] * 100))
        print("F1_Score: %.2f%%" % (result[2] * 100))
        
    ######### LSTM with k-fold cross validation #########
    def lstm_cv(self):
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        x_train_padded_data, x_test_padded_data = self.__tokenise(self.x_train, self.x_test)
        
        lstm_cv_loss = []
        lstm_cv_acc = []
        lstm_cv_f1 = []
        for train, val in kfold.split(self.x_train, self.y_train):
                  
              x_train_padded_data, x_val_padded_data = self.__tokenise(self.x_train.iloc[train], self.x_train.iloc[val])
                                   
              model = self.__lstmModel()              
              model.fit(x_train_padded_data, self.y_train.iloc[train], validation_data=(x_val_padded_data, self.y_train.iloc[val]), epochs=15, batch_size=10)
              
              # Evaluate the model
              result = model.evaluate(x_test_padded_data, self.y_test, verbose=0)
              lstm_cv_loss.append(result[0] * 100)
              lstm_cv_acc.append(result[1] * 100)
              lstm_cv_f1.append(result[2] * 100)
             
        print('Loss: ', "%.2f%%" % (np.mean(lstm_cv_loss)))
        print('Accuracy', "%.2f%%" % (np.mean(lstm_cv_acc)))
        print('F1 Score', "%.2f%%" % (np.mean(lstm_cv_f1)))
                
    def __cnnModel(self):
        model = Sequential()
        model.add(Embedding(input_dim=500, output_dim=128, mask_zero=True)) # "mask_zero=True" tell the model that data is masked, Embedding layer is used to map words into vectors
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, average=None, threshold = 0.5)])      
        print(model.summary())
        
        # generate model diagram
        plot_model(model, to_file='cnn_plot.png', show_shapes=True, show_layer_names=True)
        
        return model
        
    ######### CNN with k-fold cross validation #########
    def cnn_cv(self):
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        x_train_padded_data, x_test_padded_data = self.__tokenise(self.x_train, self.x_test)
        
        cnn_cv_loss = []
        cnn_cv_acc = []
        cnn_cv_f1 = []
        for train, val in kfold.split(self.x_train, self.y_train):
              
              x_train_padded_data, x_val_padded_data = self.__tokenise(self.x_train.iloc[train], self.x_train.iloc[val])
              
              model = self.__cnnModel()             
              model.fit(x_train_padded_data, self.y_train.iloc[train], validation_data=(x_val_padded_data, self.y_train.iloc[val]), epochs=15, batch_size=10)
              
              # Evaluate the model
              scores = model.evaluate(x_test_padded_data, self.y_test, verbose=0)
              cnn_cv_loss.append(scores[0] * 100)
              cnn_cv_acc.append(scores[1] * 100)
              cnn_cv_f1.append(scores[2] * 100)
             
        print('Loss: ', "%.2f%%" % (np.mean(cnn_cv_loss)))
        print('Accuracy', "%.2f%%" % (np.mean(cnn_cv_acc)))
        print('F1 Score', "%.2f%%" % (np.mean(cnn_cv_f1)))        
        
    ######### LSTM with hyperparameters tuning #########        
    def __lstmBuildModel(self, hp):
        
        model = Sequential()
        model.add(Embedding(input_dim=500, output_dim=hp.Int("emb_op", min_value=68, max_value=128, step=10), mask_zero=True)) # "mask_zero=True" tell the model that data is masked
       
        # Tune number of LSTM layers and it's units
        # Tune with single LSTM layer
        model.add(LSTM(hp.Int("units", min_value=32, max_value=512, step=32)))
        
        # Tune with multiple LSTM layers
        ########## Note: This is turned off since running with mulitple layers is extremely slow and has not shown improvement in performance either
        # model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32),return_sequences=True)) # , input_shape=(X_train.shape[1],X_train.shape[2]
        
        # for i in range(hp.Int('n_layers', 1, 3)):
        #     model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
        
        # model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=512,step=32)))
       
        model.add(Dense(1, activation='sigmoid'))
       
        # Tune learning rate
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, average=None, threshold = 0.5)])
        
        return model
        
    ######### CNN with hyperparameters tuning #########        
    def __cnnBuildModel(self, hp):
        model = Sequential()
        model.add(Embedding(input_dim=500, output_dim=hp.Int("emb_op", min_value=68, max_value=128, step=10), mask_zero=True)) # "mask_zero=True" tell the model that data is masked, Embedding layer is used to map words into vectors
        
        # Tune the number of Conv1D layers
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                Conv1D(
                    # Tune number of filters and kernels separately
                    filters = hp.Int(f"conv_{i}", min_value=32, max_value=512, step=32), 
                    kernel_size = hp.Int(f"conv_{i}_kernel", min_value=1, max_value=10, step= 1),
                    activation = hp.Choice("activation", ["relu", "tanh"])
                )
            )
        model.add(GlobalMaxPooling1D())
        
        # Tune the number of dense layers
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                Dense(
                # Tune number of units separately
                units=hp.Int(f"units_{i}", min_value=1, max_value=10, step=1),
                activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
        
        model.add(Dense(1, activation='sigmoid'))
        
        # Tune learning rate
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tfa.metrics.F1Score(num_classes=1, average=None, threshold = 0.5)],
        )
        return model
        
    def __modelHpTuning(self, modelName):  
       
        x_train_padded_data, x_test_padded_data = self.__tokenise(self.x_train, self.x_test)
                
        hypermodel = None
        if (modelName == 'CNN'):
            hypermodel = self.__cnnBuildModel
            
        elif (modelName == 'LSTM'):
            hypermodel = self.__lstmBuildModel
            
        tuner = keras_tuner.Hyperband(
                    hypermodel=hypermodel,
                    objective="val_accuracy",
                    max_epochs=10,
                    factor=3,
                    overwrite=True,
                    directory="results",
                    project_name="maritime",
                )
                    
        # Split training dataset into training and validation sets
        x_train2, x_val, y_train2, y_val = util.split_train_test(self.x_train, self.y_train, stratifyColumn = self.y_train)
        
        # Tokenize split training dataset
        x_train2_padded_data, x_val_padded_data = self.__tokenise(x_train2, x_val)        
              
        # Build the defined model        
        if (modelName == 'CNN'):
            self.__cnnBuildModel(keras_tuner.HyperParameters())
            
        elif (modelName == 'LSTM'):
            self.__lstmBuildModel(keras_tuner.HyperParameters())        
        
        # Callback to stop training early after reaching a certain value for the validation loss
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        
        # Search for the top hyperparameters
        tuner.search(x_train2_padded_data, y_train2, epochs=15, batch_size=10, validation_data=(x_val_padded_data, y_val), validation_split=0.2, callbacks=[stop_early])         
        print(tuner.results_summary())
        
        # Get the top hyperparameters
        best_hps=tuner.get_best_hyperparameters()[0]
        
        # Build the model with the best hp
        model = None
        if (modelName == 'CNN'):
            model = self.__cnnBuildModel(best_hps)
            
        elif (modelName == 'LSTM'):
            model = self.__lstmBuildModel(best_hps)        
        
        training_result = model.fit(x_train_padded_data, self.y_train, validation_data=(x_test_padded_data, self.y_test), epochs=15, batch_size=10)
    
        # Evaluate the best number of epochs
        val_acc_per_epoch = training_result.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch))
        
        # Retrain and evaluate the model
        model.fit(x_train_padded_data, self.y_train, validation_data=(x_test_padded_data, self.y_test), epochs=best_epoch, batch_size=10)
        retraining_result = model.evaluate(x_test_padded_data, self.y_test, verbose=True)
        print("%s: %.2f%%" % (model.metrics_names[0], retraining_result[0]*100))
        print("%s: %.2f%%" % (model.metrics_names[1], retraining_result[1]*100))
        print("%s: %.2f%%" % (model.metrics_names[2], retraining_result[2]*100))
        
    def cnn_hp_tuning(self):
        self.__modelHpTuning('CNN')
        
    def lstm_hp_tuning(self):
        self.__modelHpTuning('LSTM')

if __name__ == '__main__':
    # It is recommended to execute one model at a time to avoid memory issues.
    # LSTM models
    Models().lstm_hold_out() # accuracy results: 86.67%
    Models().lstm_cv() # mean accuracy results: 83.00%
    Models().lstm_hp_tuning() # accuracy results: 86.67% with single LSTM layer and accuracy: 80.00% with multiple LSTM layers  
    
    # CNN models
    Models().cnn_cv() # mean accuracy results: 81.00%
    Models().cnn_hp_tuning() # accuracy results: 93.33%
    
    