 #####################################################
#    Script to train the NN for the univariate case   #
#                    By: Diana Diaz                   #
 #####################################################

#This neural network has as inputs albedo values from 0 to 1 with 0.1 steps and the output to train the NN is the fnt_sw_toa variable from RRTMG model

### Import libraries
## general libraries 
import os,sys,glob
import numpy as np
import pandas as pd
import xarray as xr
import h5py

# for plots
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

## ML libraries
# tensorflow
import tensorflow as tf
from tensorflow import keras
#print('tensorflow version:', tf.__version__) 

# keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from keras.utils.vis_utils import plot_model #CHANGE
#from keras.utils import plot_model #LAPTOP
import keras.backend as K
from tensorflow.python.training import checkpoint_utils as cp

#Extra libraries
from ann_visualizer.visualize import ann_viz
from itertools import chain
from mpl_toolkits import mplot3d
from sympy import symbols 
from sympy.utilities.lambdify import lambdify
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import time
import matplotlib as mpl

print(f'Using Python  = {sys.version.split()[0]}')
print(f'Tensorflow    = {tf.__version__}')
print(f'Keras Version = {keras.__version__}')

## set the font size globally
font_size = 12
mpl.rcParams['font.size'] = font_size

#### Import functions
sys.path.append('../')
from myFunctions import read_netcdfs
from myFunctions import build_nn_eq
from myFunctions import runNN
from myFunctions import MinMaxNorm
from myFunctions import MinMaxInverse

### Import data 
#We will be using data from RRTMG model

##### RRTMG 
#RRTMG data is used as output variable

##### baseline profiles 

'''Modify path if necessary'''
bsln = xr.open_dataset('/storage/ddiaz/2023/thesis/rrtmg/output_files/2015-06-01T00_lat84_lon17-5/albPerturbation/toa_sfc/DD_2015-06-01T00_lat84_lon17-5_a06_t1_toa_sfc.nc').to_dataframe()

#####  perturbed albedo profile

prtrbd = read_netcdfs('/storage/ddiaz/2023/thesis/rrtmg/output_files/2015-06-01T00_lat84_lon17-5/albPerturbation/toa_sfc/DD*_a*toa_sfc.nc', dim='time').to_dataframe()
prtrbd= prtrbd.reset_index(drop=True)

##### albedo

albedo=pd.DataFrame(np.arange(0,1.1,0.1),columns=['albedo'],index=None)

#dataframe with one timestep of the values of net flux of sw at TOA with its corresponding albedo values
df = albedo.merge(prtrbd.fnt_sw_toa, left_index=True, right_index=True)

cc = [0.8]*11
cc=pd.DataFrame(cc,columns=['cc'],index=None)

# we combine the input variables
merged_array_nl = np.stack((albedo, cc), axis=1)

### scale inputs and outputs

x = np.array(df.albedo).reshape(-1, 1)
y = df.fnt_sw_toa

scale_x = MinMaxScaler()
scale_y = MinMaxScaler()

x = scale_x.fit_transform(x)

y = y.values.reshape(len(y),1)
y = scale_y.fit_transform( y )

### 1 hidden layer, 2 nodes, tanh activation function

tf.random.set_seed(6)

lr = 0.001
itrtns = 90000
shp_inpt = 1
ndsLyr1 =2
ndsLyr2 =0
act_fnctn1 = 'tanh'
act_fnctn2 = 0

model,history_df,weights_dict = runNN(x, y,
                               alpha=lr,
                               iterations=itrtns,
                               shape_input=shp_inpt,
                               nodesLayer1=ndsLyr1,
                               nodesLayer2=ndsLyr2,
                               act_function1=act_fnctn1,
                               act_function2=act_fnctn2)

model.get_weights()

#### Function to build the NN equation from the weight and biases outputs

nds_num = ndsLyr1
inpt_num = shp_inpt
inpt_list = ['a','cc','a_cc']
act_func = 'tanh'

# temp list as a container for each layer formulas
formula_list = []
frml_eqn = []
y_str = []

equation = build_nn_eq(nds_num,inpt_num_inpt_list,act_func,model)

#predictions
x2 = np.array(df.albedo).reshape(-1, 1)
x2 = scale_x.fit_transform(x2)
prdctns=model.predict(x2)

#### inverse transforms
prdctns_plot = scale_y.inverse_transform(prdctns)

plt.figure(figsize=(6,6),dpi=300)

plt.plot(df.albedo,df.fnt_sw_toa,marker='o',color='g',markersize=2.5,label='real values')
plt.scatter(x=df.albedo,y= prdctns_plot,marker='.',facecolors='none', edgecolors='b',s=200,label='NN predictions')

plt.xlabel('surface albedo')
plt.ylabel('Radiative flux at TOA $ [W/m^2] $')
#plt.title('Interpolation')
plt.legend()
plt.savefig('scatterplot.png')

### Albedo feedback 
##### NN 

baseline=float(prdctns_plot[np.where(np.round(df.albedo,2)==0.6)])

deltaNN = []
for ii in prdctns_plot:
    val=float(ii)-baseline
    deltaNN.append(val)

##### RRTMG

deltaRa=df.fnt_sw_toa.values-bsln.fnt_sw_toa.values #data.mean(axis=1).values-t.fnt_sw_toa.mean().values

### Kernel 

### import file
path = '/storage/ddiaz/2023/thesis/kernels_han/'
fl_2012 = 'RRTM_kernel_2012_cld_alb_TOA_SFC_09.nc'
fl_2013 = 'RRTM_kernel_2013_cld_alb_TOA_SFC_09.nc'
fl_2015 = 'RRTM_kernel_2015_cld_alb_TOA_SFC_06.nc'

krnl_2015 = xr.open_dataset(path+fl_2015)

toa_2015_06 = krnl_2015['TOA'].sel(up_down_net=3 , band = 1)

a=np.arange(0,1.1,0.1)
for ii in a:
    delta_a = a-0.6

deltaR_Kx = toa_2015_06.sel(latitude=85.0,longitude=17.5).values*delta_a

a=np.arange(0,1.1,0.1)

plt.figure(figsize=(6, 8), dpi=300) #80


plt.scatter(df.albedo,deltaNN,marker='.',s=200,facecolors='none', edgecolors='b',label='NN') #color='g',markersize=20
plt.plot(a,deltaRa.squeeze(),marker= 'o',color='g',markersize=2.5,label='RRTMG')


plt.plot(a,deltaR_Kx,color='darkgoldenrod',label='Kernel',markersize=10)
plt.axvline(x=0.6,linestyle = '--',color='k')
plt.axhline(y=0, linestyle = '--',color='k')
plt.ylabel('TOA $\Delta [W/m^2] $')
#$ [W/m^2] $
plt.xlabel('surface albedo')
plt.title('\n$\Delta$R = R(a,$x_{2015}$)-R($a_{2015}$,$x_{2015}$)  \nlat= 84$^\circ$, lon=17.5$^\circ$, 2015-06-01 00:00 UTC')

plt.text(0.65, 10, 'a = 0.6', fontsize=13)

lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=13)
lgnd.legendHandles[0]._sizes = [60]
lgnd.legendHandles[1]._sizes = [60]
lgnd.legendHandles[2]._sizes = [60]

plt.savefig('univariate_fig.png')
#plt.show()
