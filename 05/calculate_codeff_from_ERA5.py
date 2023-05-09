## general libraries 
import os,sys,glob
import numpy as np
import pandas as pd
import xarray as xr

import cfgrib
import h5py

# for plots
%matplotlib inline
import matplotlib.pyplot as plt

##### Upload datasets 

path = '/home/diladiga/projects/rrg-yihuang-ad/diladiga/datasets/' #'//storage/ddiaz/2023/myProject/data'
file_pl= 'era5_monthly_avg_pl_1990_2020_cc_ciwc_clwc.grib'

file_sl = 'era5_monthly_avg_sl_1990-2020_tisr_tciw_tclw_tcwv_hcc_mcc_lcc_sp_tco3_fal_tsr_msl.nc'

xrr_pl = xr.open_dataset(path+file_pl,engine='cfgrib')

xrr_pl = xrr_pl.sel(time=slice('2015','2020'))

p = xrr_pl.isobaricInhPa.values

#### pressure levels

dp = []
for ii in np.arange(len(p)-1):
    DP = p[ii]-p[ii+1]
    dp.append(DP)

del DP

#We add one value to dP so every level of pressure has a dP value
dp.insert(0,25)   

#### Optical depth 
effradliq = 10
effradice = 25
H=9000

data_arrays_liq = []

for ii in np.arange(18):
    P = p[ii]
    DP = dp[ii]
    xrr = xrr_pl.sel(isobaricInhPa = P) #if netcdf file, use level; if grib file use isobaricInhPa
    tauliq_c = (3/2) * (1*xrr.clwc/effradliq)*(DP/P*H)
    data_arrays_liq.append(tauliq_c)

xrr_tauliq = xr.concat(data_arrays_liq,dim='level')

del data_arrays_liq

data_arrays_ice = []

for ii in np.arange(18):
    P = p[ii]
    DP = dp[ii]
    xrr = xrr_pl.sel(isobaricInhPa = P) #if netcdf file, use level; if grib file use isobaricInhPa
    tauice_c = (3/2) * (1*xrr.ciwc/effradice)*(DP/P*H)
    data_arrays_ice.append(tauice_c)

xrr_tauice = xr.concat(data_arrays_ice,dim='level')

del data_arrays_ice

#### sumar optical depths de todos los levels 

efftauliq_sum = (xrr_tauliq*xrr_pl.cc).sum(dim='isobaricInhPa')
efftauice_sum = (xrr_tauice*xrr_pl.cc).sum(dim='isobaricInhPa')


efftau = efftauliq_sum + efftauice_sum

#### we save this variable

efftau.to_netcdf(path = path+'/monthly_avg_sl_2015-2020_efftau.nc', mode='w')
