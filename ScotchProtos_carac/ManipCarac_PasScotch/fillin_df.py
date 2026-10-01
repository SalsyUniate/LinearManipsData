import os
import re
from math import sqrt
from os.path import dirname
from os.path import join as pjoin

import numpy as np
import pandas as pd
import scipy.io as sio


dirname = '2026_5_11_11h32'

#### Get header from one file ####
mat_fname = '2026_5_11_11h32/acq_00001_28_Hz_5000_Ohms_1_ms2_1_sweep.mat'
mat_contents = sio.loadmat(mat_fname, spmatrix=False) # get file contents

headers = list(mat_contents.keys()) # get file headers
headers.append('power') # add columns
headers.append('v_rms')
headers.append('x_rms')
headers.append('dotx_rms')
headers.append('ddotx_rms')
dataframe = pd.DataFrame(columns = headers) # create dataframe from headers list 
print(dataframe)


def makeline(index, data):
                
    # computing power and rms values for signals 
    
    vmoy = np.mean(data['tension_meas_vect'][0])
    vpasmoy = data['tension_meas_vect'][0] - vmoy
    v_rms = sqrt(np.mean(vpasmoy**2))
    
    xmoy = np.mean(data['dep_meas_vect'][0])
    xpasmoy = data['dep_meas_vect'][0] - xmoy
    x_rms = sqrt(np.mean(xpasmoy**2))
    
    dotxmoy = np.mean(data['vit_meas_vect'][0])
    dotxpasmoy = data['vit_meas_vect'][0] - dotxmoy
    dotx_rms = sqrt(np.mean(dotxpasmoy**2))
    
    ddotxmoy = np.mean(data['acc_meas_vect'][0])
    ddotxpasmoy = data['acc_meas_vect'][0] - ddotxmoy
    ddotx_rms = sqrt(np.mean(ddotxpasmoy**2))
    
    R = data['Rload'][0][0]
    calc_power = v_rms**2/(R)
    
    # fill in dataframe line 
    
    dataframe.loc[index] = [data['__header__'],
                            data['__version__'],
                            data['__globals__'],
                            R,
                            data['acc_amp'][0][0],
                            np.mean(data['acc_meas_vect'][0]),
                            np.mean(data['dep_meas_vect'][0]),
                            data['f'][0][0],
                            data['sweep_up'][0][0],
                            np.mean(data['tension_meas_vect'][0]),
                            np.mean(data['time_meas'][0]),
                            np.mean(data['vit_meas_vect'][0]),
                            calc_power,
                            v_rms, 
                            x_rms,
                            dotx_rms,
                            ddotx_rms
                            ]
    return None


#### fill in dataframe ####


for i in range(1,10):
    pattern = f'acq_0000{i}'
    generic_name = re.compile(pattern)
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if generic_name.match(file):
                newdata = sio.loadmat(pjoin(dirname,file), spmatrix=False)
                makeline(i, newdata)
                break

for i in range(10,100):
    pattern = f'acq_000{i}'
    generic_name = re.compile(pattern)
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if generic_name.match(file):
                newdata = sio.loadmat(pjoin(dirname,file), spmatrix=False)
                makeline(i, newdata)
                break

for i in range(100,1000):
    pattern = f'acq_00{i}'
    generic_name = re.compile(pattern)
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if generic_name.match(file):
                newdata = sio.loadmat(pjoin(dirname,file), spmatrix=False)
                makeline(i, newdata)
                break

for i in range(1000,1501):
    pattern = f'acq_0{i}'
    generic_name = re.compile(pattern)
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if generic_name.match(file):
                newdata = sio.loadmat(pjoin(dirname,file), spmatrix=False)
                makeline(i, newdata)
                break

print(dataframe.shape)