 ####################################################
#    Script to train the NN for the bivariate case   #
#                    By: Diana Diaz                  #
 ####################################################

'''
This neural network has as inputs albedo and cloud cover values from 0 to 1 with 0.1 steps 
and the output to train the NN is the net flux of shortwave radiation at TOA (fnt_sw_toa) 
variable from the RRTMG model
'''

### Import libraries
## general libraries 
import os,sys,glob
import numpy as np
import pandas as pd
import xarray as xr
import h5py

# plots
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

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
import scipy
from scipy.interpolate import RegularGridInterpolator
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

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

### Import data 
#We will be using data from RRTMG model

##### RRTMG 
#RRTMG data is used as output variable


##### baseline profiles 

'''Modify path if necessary'''
bsln = xr.open_dataset('/storage/ddiaz/2022/myProject/august/rrtmg/01062016_prtrbd_a_cc/01062016_toa_sfc.nc').to_dataframe()

#####  perturbed albedo profile

prtrbd = read_netcdfs('/storage/ddiaz/2023/thesis/rrtmg/output_files/2015-06-01T00_lat84_lon17-5/alb_cc_Perturbation/toa_sfc/DD*_a*toa_sfc.nc', dim='time').to_dataframe()
prtrbd = prtrbd.reset_index(drop=True)

#### Join datasets

#we add the albedo and cloud cover variables to our dataFrame called prtrbd
aa=np.round(np.arange(0,1.1,0.1),2)
albedo =  list(chain.from_iterable(zip(*[ aa ]*11))) 
cc = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]*11

prtrbd['albedo'] = albedo
prtrbd['cc'] = cc

random1 = prtrbd.sample(frac=1) # we shuffle the data

### Two input variables

##### merged array
df=pd.DataFrame(prtrbd.fnt_sw_toa,columns=['fnt_sw_toa'])
df['albedo'] = prtrbd.albedo
df['cc'] = prtrbd.cc

#### plot
plt.figure(figsize=(6,6),dpi=300)
ax = plt.axes(projection='3d')
#ax.scatter3D(random.albedo,random.cc,random.fnt_sw_toa,s=0.1,color='gray') #label='RRTMG interpolated values',
ax.scatter3D(df.albedo,df.cc,df.fnt_sw_toa,label='RRTMG values',s=5,color= 'navy')

#plt.title('fnt_sw_toa(albedo,cc) \n Scale inputs and outputs \n 1 hidden layer, 2 nodes \n tanh act func')
plt.xlabel('surface albedo')
plt.ylabel('cloud cover')

lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=13)
lgnd.legendHandles[0]._sizes = [30]
#lgnd.legendHandles[1]._sizes = [30]

ax.view_init(5, 333) #333
plt.savefig('sfcAlb_cc_flux.png')

###  we add an input variable wich is the multiplication of albedo and cloud cover (a*cc)
a_cc = df.albedo*df.cc

merged_array = np.stack((df.albedo, df.cc,a_cc), axis=1)
y = df.fnt_sw_toa    

### Training and testing with all data
xx = merged_array
yy = y

#Comment these linese if no normalization is needed!!
xx = MinMaxNorm(merged_array,-1,1) #scale_x.fit_transform(xx)
yy = yy.values.reshape(len(yy),1)
yy = MinMaxNorm(yy,-1,1)#scale_y.fit_transform( yy )

tf.random.set_seed(6)
lr = 0.001
itrtns = 90000
shp_inpt = 3
ndsLyr1 =2
ndsLyr2 =0
act_fnctn1 = 'tanh'
act_fnctn2 = 0
ptnce = 100

model,history_df,weights_dict = runNN(xx, yy,
                                    alpha=lr,
                                    iterations=itrtns,
                                    shape_input=shp_inpt,
                                    nodesLayer1=ndsLyr1,
                                    nodesLayer2=ndsLyr2,
                                    act_function1=act_fnctn1,
                                    act_function2=act_fnctn2,
                                    patienceEpochs=ptnce)

len(history_df.values)

model.trainable_weights

#### Function to build the NN equation from the weight and biases outputs
nds_num = ndsLyr1
inpt_num = shp_inpt
inpt_list = ['a','cc','a_cc']
act_func = 'tanh'

equation = build_nn_eq(nds_num,inpt_num_inpt_list,act_func,model)

#predictions
xx2 = MinMaxNorm(merged_array,-1,1)
prdctnss=model.predict(xx2)

# inverse transforms
prdctns_plott = MinMaxInverse(prdctnss,-1,1,df.fnt_sw_toa)#scale_y.inverse_transform(prdctnss)

I = df.albedo #a_new
II = df.cc #cc_new
III = df.fnt_sw_toa #z_interp

plt.figure(figsize=(6,6),dpi=300)
ax = plt.axes(projection='3d')
ax.scatter3D(I,II,III,label='RRTMG values',s=0.1)
ax.scatter3D(I,II,prdctns_plott,label='NN predictions',s=0.1)
plt.legend(loc='upper right')
#plt.title('fnt_sw_toa(albedo,cc) \n Scale inputs and outputs \n 1 hidden layer, 2 nodes \n tanh act func')
plt.xlabel('albedo')
plt.ylabel('cc')
ax.view_init(5, 333) #333
plt.savefig('scatter_plot.png')

### albedo feedback 

##### NN 
baseline=df.fnt_sw_toa[np.where(df.albedo==0.6)[0][5]] # and where cc = 0.5

deltaNN = []
for aa in df.fnt_sw_toa:
    val=float(aa)-baseline
    deltaNN.append(val)

##### RRTMG
deltaRa=df.fnt_sw_toa - baseline

#####  Plot of the univariate albedo feedback at the TOA measured by the NN and RRTMG
a=np.arange(0,1.1,0.1)
plt.figure(figsize=(6, 8), dpi=300) #80
plt.axvline(x=0.6,linestyle = '--',color='k')
plt.axhline(y=0, linestyle = '--',color='k')
#plt.plot(dataset.albedo,deltaNN,color='b',label='NN')
plt.scatter(df.albedo,deltaNN,facecolors='none', edgecolors='b', marker = '.',s=200, label='NN')

plt.plot(df.albedo[np.where(df.cc==0)[0]],deltaRa[np.where(df.cc==0)[0]] ,color='g',marker = 'o',markersize=2.5,label='RRTMG')
plt.plot(df.albedo[np.where(df.cc==0.1)[0]],deltaRa[np.where(df.cc==0.1)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.2)[0]],deltaRa[np.where(df.cc==0.2)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.3)[0]],deltaRa[np.where(df.cc==0.3)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.4)[0]],deltaRa[np.where(df.cc==0.4)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.5)[0]],deltaRa[np.where(df.cc==0.5)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.6)[0]],deltaRa[np.where(df.cc==0.6)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.7)[0]],deltaRa[np.where(df.cc==0.7)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.8)[0]],deltaRa[np.where(df.cc==0.8)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==0.9)[0]],deltaRa[np.where(df.cc==0.9)[0]] ,color='g',marker = 'o',markersize=2.5)
plt.plot(df.albedo[np.where(df.cc==1.0)[0]],deltaRa[np.where(df.cc==1.0)[0]] ,color='g',marker = 'o',markersize=2.5)

#plt.scatter(df.albedo,deltaRa.squeeze(),color='g', marker = 'o',s=8, label='RRTMG')


plt.text(0.69, 10, 'a = 0.6 and cc=0.5', fontsize=13)

plt.ylabel('TOA 'r'$\Delta$R ('r'$Wm^{-2}$)')
plt.xlabel('surface albedo')
plt.title('$\Delta$R = R(a,c,$x_{2015}$)-R($a_{2015}$,$c_{2015}$,$x_{2015}$)  \nlat= 84$^\circ$, lon=17.5$^\circ$, 2015-06-01 00:00 UTC')
#plt.title(''r'$\Delta$R = R(a,'r'$x_{2016}$)-R('r'$a_{2016}$,'r'$x_{2016}$)')

lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=13)
lgnd.legendHandles[0]._sizes = [120]
lgnd.legendHandles[1]._sizes = [30]

plt.savefig('bivariate_fig.png')
