 ################################################
#    Script to train the global neural network   #
#                 By: Diana Diaz                 #
 ################################################

## import general libraries 
import os,sys,glob
import numpy as np
import pandas as pd
import xarray as xr
import math
import time

## import libraries for plots
import matplotlib.pyplot as plt
import matplotlib as mpl

# sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

## ML libraries
# tensorflow
import tensorflow as tf
from tensorflow import keras
# keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from keras.utils.vis_utils import plot_model #CHANGE
#from keras.utils import plot_model #LAPTOP
import keras.backend as K
from tensorflow.python.training import checkpoint_utils as cp

## extra libraries
from ann_visualizer.visualize import ann_viz
from itertools import chain
from mpl_toolkits import mplot3d
from sympy import symbols 
from sympy.utilities.lambdify import lambdify

#import functions
from myFunctions import read_netcdfs
from myFunctions import build_nn_eq
from myFunctions import runNN
from myFunctions import MinMaxNorm
from myFunctions import MinMaxInverse

## we define the parameters for the NN configuration
lr = 0.001
itrtns = 900
ndsLyr1 =11
ndsLyr2 =0
act_fnctn1 = 'tanh'
act_fnctn2 = 0
earlyStppng = 1                        # No = 0 , Yes = 1
ptnce = 10                            #100
lossTHR = 0.00002                      #0.00003

## set the font size globally
font_size = 12
mpl.rcParams['font.size'] = font_size

## import data
# single levels data including effective cloud optical depth variable
#path = '/home/diladiga/projects/rrg-yihuang-ad/diladiga/NeuralNetwork/thesis/global_nn_model/ecod_calculation/'
path = '../data/'
file_sl = 'era5_1deg_monthly_avg_sl_1990-2020_tisr_tciw_tclw_tcwv_hcc_mcc_lcc_sp_tco3_fal_tsr_msl_ecod_noZeros.nc'
xrr_sl = xr.open_dataset(path+file_sl)

#To convert to W m-2 , the accumulated values should be divided by the accumulation period expressed in seconds
xrr_sl['tisr'] = xrr_sl.tisr/24/3600
xrr_sl['tsr'] =xrr_sl.tsr/24/3600

## divide datasets for training and testing 
# extract the years from the time dimension
years = xrr_sl['time.year']

# select odd years
#odd years for testing
xrr_sl_odd = xrr_sl.sel(time=(years % 2 == 1))

# select even years
#even years for trainning
xrr_sl_even = xrr_sl.sel(time=(years % 2 == 0))

df_train = xrr_sl_even.to_dataframe()

#### Input variables
vars_list = [np.array(df_train.tisr), np.array(df_train.tciw), np.array(df_train.tclw), np.array(df_train.tcwv), 
             np.array(df_train.lcc), np.array(df_train.mcc), np.array(df_train.hcc), np.array(df_train.sp),
             np.array(df_train.tco3), np.array(df_train.fal), np.array(df_train.ecod), 
             np.array(df_train.ecod)*np.array(df_train.fal)
            ]

vars_list_str = ['tisr', 'tciw', 'tclw', 'tcwv', 'lcc', 'mcc', 'hcc', 'sp', 'tco3' , 'fal','ecod','ecod_alb']

#### Output variable
tsr = np.array(df_train.tsr)
max_val = np.round(np.max(tsr),2)
min_val = np.round(np.min(tsr),2)

# Normalization range [a,b]
#We set the normalization rate to be between -1 and 1, since the activation function (tanh) has the same range
a = -1
b = 1

nrmlzd_vars_list = []
for x in vars_list:    
    n = MinMaxNorm(x,a,b)
    nrmlzd_vars_list.append(n)

#we combine the input variables into one merged array
merged_array = np.stack((nrmlzd_vars_list), axis=1) #da.stack((nrmlzd_vars_list), axis=1)

### we define here the input and output variables for the NN
xx = merged_array   #the input variables are already normalized
yy = MinMaxNorm(tsr,a,b) #we normalize the output

#### new early stopping method

from keras.callbacks import Callback

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current <= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


### Training
tf.random.set_seed(6)
shp_inpt = merged_array.shape[1]

model,history_df,weights_dict = runNN(xx, yy,
                                    alpha=lr,
                                    iterations=itrtns,
                                    shape_input=shp_inpt,
                                    nodesLayer1=ndsLyr1,
                                    nodesLayer2=ndsLyr2,
                                    act_function1=act_fnctn1,
                                    act_function2=act_fnctn2,
                                    earlyStopping=earlyStppng,
                                    patienceEpochs=ptnce,
                                    loss_thr=lossTHR ) #we define the loss threshold but we are not using it

model.save('nn_saved_model')
nds_num = ndsLyr1
inpt_num = shp_inpt
inpt_list = vars_list_str
act_func = 'tanh'

equation = build_nn_eq(nds_num,inpt_num,inpt_list,act_func,model)
