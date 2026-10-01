import numpy as np
from fillin_df import dataframe

dataframe.sort_values(["f", "Rload"], inplace =True)
f_count = dataframe['f'].value_counts()
r_count = dataframe['Rload'].value_counts()
frequency_list = list(f_count.index)
frequency_list.sort()
resistance_list = list(r_count.index)
resistance_list.sort()
power_list = np.zeros((len(resistance_list),len(frequency_list)))
dep_list = np.zeros((len(resistance_list),len(frequency_list)))
tens_list = np.zeros((len(resistance_list),len(frequency_list)))
vit_list = np.zeros((len(resistance_list),len(frequency_list)))
acc_list = np.zeros((len(resistance_list),len(frequency_list)))

for i, f in enumerate(frequency_list):
    for j, r in enumerate(resistance_list):
        small_df = dataframe[(dataframe['f'] == f) & (dataframe['Rload'] == r)]
        
        power_list[j][i]    = small_df['power'].iloc[0]
        dep_list[j][i]      = small_df['x_rms'].iloc[0]
        tens_list[j][i]     = small_df['v_rms'].iloc[0]
        vit_list[j][i]      = small_df['dotx_rms'].iloc[0]
        acc_list[j][i]      = small_df['ddotx_rms'].iloc[0]


Pmax_over_f = power_list.max(axis=0)