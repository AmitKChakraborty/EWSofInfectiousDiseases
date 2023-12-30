#Test simulations of Flu data of the UK. 
#Some parts of this code are Edited from published codes by Bury et al. (2021), Deep learning for early warning signals of tipping points, PNAS. 
#python libraries
import numpy as np 
import pandas as pd 
import matplotlib as mlp 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import ewstools 
import random

df_import = pd.read_csv('../data/empirical/output-EpiEstim-flu-UK.csv')
region = 'UK'

df_import['Time'] = range(0, len(df_import))
df_import.set_index(['Time'], inplace=True)
print('df_import=', df_import)

df = df_import.copy()
# Replace NA with zero
df = df.fillna(0)

current_series = None
tem_time_count = []
all_series = []
k = 1

#slicing series based on Re<1
for i in range(len(df)):
    if df['Mean.R.'][i] < 1:
        tem_time_start = int(df['t_start'][i])
        tem_time_end = int(df['t_end'][i])
        tem_time_count.extend([tem_time_start, tem_time_end])
        current_series = df.loc[tem_time_count[0]-1:tem_time_count[-1]-1,['output.I']].copy()
        current_series.columns = [f'series-{k}']
    else:
        if current_series is not None:
            all_series.append(current_series)
            current_series = None 
            tem_time_count = []
            k += 1

if current_series is not None:
    all_series.append(current_series)

all_series_index_reset = [df.reset_index() for df in all_series]
combined_sliced_series = pd.concat(all_series_index_reset, axis=1)
print('combined_sliced_series=',combined_sliced_series)

sorted_sliced_series_num = 1
all_sorted_sliced_series = []

for j in range(1, k):
    series = 'series-'+ str(j)
        
    df_short = combined_sliced_series[[series]].dropna()
    df_short.columns = ['I']

    df_cases = df_short[['I']].reset_index(drop=True)
    df_cases['Time'] = range(0, len(df_cases))
    df_cases.set_index(['Time'], inplace=True)
    df_cases = df_cases.copy()
    
    if len(df_cases) >= 30: 
        # rolling window
        rw = 0.25                                    
        # span for Lowess smoothing
        span=0.2                                     
        # autocorrelation lag times
        lags = [1]                                   
        ews = ['var','ac']
        var = 'I'
        
        # set up a list to store output dataframes from ews_compute- we will concatenate them at the end
        appended_ews = []
        appended_pspec = []

        print('\nBegin EWS computation\n')
        for var in [var]:
            tem_series = df_cases[var]                            
            ews_dic = ewstools.core.TimeSeries(tem_series, 
                                              transition=None)
            ews_dic.detrend(method='Lowess', span=span)                                #dtrending data
            ews_dic.compute_auto(rolling_window=rw, lag=1)                             #computing AC
            ews_dic.compute_var(rolling_window=rw)                                     #computing VAR

            # The DataFrame of EWS
            z1 = ews_dic.state                                                         #storing state variabls of EWS
            z2 = ews_dic.ews                                                           #storing ac1, var of EWS
            df_ews_temp = pd.concat([z1, z2], axis=1)                                  #marging state variables, ac1, var 

            # Include a column in the DataFrames for realisation number and variable
            # df_ews_temp['tsid'] = i+1
            df_ews_temp['Variable'] = var
            
            df_ews_temp['tsid'] = sorted_sliced_series_num

            # Add DataFrames to list
            appended_ews.append(df_ews_temp)

            # # Print status every realisation
            # if np.remainder(i+1,1)==0:
            print('EWS for realisation complete' + str(j))

        # Concatenate EWS DataFrames. Index [tsid, Variable, Time]
        df_ews = pd.concat(appended_ews).reset_index().set_index(['tsid','Variable','Time'])
        
        all_sorted_sliced_series.append(df_ews)
        all_sorted_sliced_series_dataframe = pd.concat(all_sorted_sliced_series)


        df_resids = df_ews.reset_index()[['Time','residuals']]
        filepath='../data/resids/resids_Flu_UK_forced{}.csv'.format(sorted_sliced_series_num)
        df_resids.to_csv(filepath,
                         index=False)

        #filepath='../data/resids/df_Flu_{}{}--{}.csv'.format(region,sorted_sliced_series_num,j)
        #df_ews.to_csv(filepath,
        #                 index=True)
        
        filepath='../data/ews/df_ews_Flu.csv'.format()
        all_sorted_sliced_series_dataframe.to_csv(filepath,
                                                  index=True)
        sorted_sliced_series_num += 1
    

print('all_sorted_sliced_series_dataframe=', all_sorted_sliced_series_dataframe)
