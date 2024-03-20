#Training data generation of SIR model with Demographic Noise
#Edited from published codes by Bury et al. (2021), Deep learning for early warning signals of tipping points, PNAS. 
#python libraries
import numpy as np 
import pandas as pd 
import matplotlib as mlp 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import ewstools 
import random
import sys

#parameters
dt = 0.01
t0 = 0
tmax = 1500
tburn = 100                                 
numSims = 1000                         
seed = 0                                    

dt2 = 1                                      
rw = 0.25                                    
span=0.2                                     
lags = [1]                                 
ews = ['var','ac']

# Model
def de_fun_S(S,I,Lambda,beta,mu):
  return Lambda-beta*S*I-mu*S

def de_fun_I(S,I,beta,alpha,mu):
  return beta*S*I- alpha*I-mu*I


#batches
# count = int(sys.argv[1])  
count = 1

#model parameters
Lambda = 100                                          
alpha = 1
mu = 1
S0 = 500                                   
I0 = 7

# bifurcation point
betabif = mu*(alpha+mu)/Lambda                   
print(betabif)

# DataFrame for each variable
df_sims_S = pd.DataFrame([])                     
df_sims_I = pd.DataFrame([])

#arrays to store single time-series data
t = np.arange(t0,tmax,dt)                        
S = np.zeros(len(t))                            
I = np.zeros(len(t))

beta = np.zeros(len(t)) 
beta_intercept = []
beta_slope = []
tbif = np.zeros(numSims) 

right_intercept = betabif/2 
mid_intercept = right_intercept/2 

beta_intercept = np.random.triangular(0, mid_intercept, right_intercept, numSims)

for j in range(numSims // 2):
    left_slope = 0
    right_slope = (betabif-beta_intercept[j])/1500
    mid_slope = right_slope/2
    slope = np.random.triangular(left_slope, mid_slope, right_slope)
    beta_slope.append(slope)

for j in range(numSims // 2, numSims):
    left_slope = (betabif-beta_intercept[j])/1500
    right_slope = (2*betabif-beta_intercept[j])/1500
    mid_slope = right_slope/2
    slope = np.random.triangular(left_slope, mid_slope, right_slope)
    beta_slope.append(slope) 

print(beta_intercept)
print(beta_slope)

 # Initialise a list to collect trajectories
list_traj_append = []                                                            
label_list = []
tbif = np.zeros(numSims) 

# loop over simulations
print('\nBegin simulations \n')
for j in range(numSims):                
    # noise intensity
    sigma_S = np.random.triangular(0, 0.5, 1)                                    
    sigma_I = np.random.triangular(0, 0.5, 1)
    
    beta0 = beta_intercept[j]
    beta1 = beta_slope[j]
    beta = np.zeros(len(t))

    for k in range(len(t)): 
        beta[k] = beta0 + t[k]*beta1

    beta = pd.Series(beta, index=t) 
    
    # Time at which bifurcation occurs; we set bifurcation point at 1500 for null simulations
    if betabif <= beta.iloc[len(t)-1]:
        tbif[j] = beta[beta > betabif].index[1] 
    else:
        tbif[j] = 1500
    print(tbif)
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_S_burn = np.random.normal(loc=0, scale=sigma_S*np.sqrt(dt), size = int(tburn/dt))   
    dW_S = np.random.normal(loc=0, scale=sigma_S*np.sqrt(dt), size = len(t))
    
    dW_I_burn = np.random.normal(loc=0, scale=sigma_I*np.sqrt(dt), size = int(tburn/dt))
    dW_I = np.random.normal(loc=0, scale=sigma_I*np.sqrt(dt), size = len(t))
    
    # Run burn-in period
    for i in range(int(tburn/dt)):                                              
        S0 = S0 + de_fun_S(S0,I0,Lambda,beta[0],mu)*dt + dW_S_burn[i]            
        if np.isnan(S0):
            S0 = random.uniform(0, 500)
        I0 = I0 + de_fun_I(S0,I0,beta[0],alpha,mu)*dt + dW_I_burn[i]
        if np.isnan(I0): 
            I0 = random.uniform(0, 0.5)
    
    #if intial infected less than 0.1; set random initial between (0.1, 1)        
    if I0 < 0.1 :
        I0 = random.uniform(0.1 , 1)
        
    # Initial condition post burn-in period
    S[0]=S0
    I[0]=I0
    
    # Run simulation
    for i in range(len(t)-1):                                                     
        a = Lambda + beta.iloc[i]*S[i]*I[i] + mu*S[i]
        b = -beta.iloc[i]*S[i]*I[i]
        c = beta.iloc[i]*S[i]*I[i] + alpha*I[i] + mu*I[i]
        d = np.sqrt(a*c-b**2)
        e = np.sqrt(a+c+2*d)
        S[i+1] = S[i] + de_fun_S(S[i],I[i],Lambda,beta.iloc[i],mu)*dt + ((a+d)/e)*dW_S[i] + (b/e)*dW_I[i]    
        I[i+1] = I[i] + de_fun_I(S[i],I[i], beta.iloc[i],alpha,mu)*dt + (b/e)*dW_S[i] + ((c+d)/e)*dW_I[i]
        # make sure that state variable remains >= 0
        if S[i+1] < 0:                                                                                      
            S[i+1] = 0
        if I[i+1] < 0:
            I[i+1] = random.uniform(0.1 , 1)
    # Store series data in a temporary DataFrame
    data = {'tsid': (j+1)*np.ones(len(t)),                                       
                'Time': t,
                'S': S,
                'I': I}
    df_temp = pd.DataFrame(data)                                                 
    list_traj_append.append(df_temp)     
    
    if betabif <= beta.iloc[len(t)-1]:
        df_label = pd.DataFrame([1])
    else:
        df_label = pd.DataFrame([0])
    
    label_list.append(df_label)
    print('Simulation '+str(j+1)+' complete')
    
    beta0 = []
    beta1 = []

label = pd.concat(label_list, ignore_index=True)
df_traj = pd.concat(list_traj_append)                                            
df_traj.set_index(['tsid','Time'], inplace=True)

# Compute EWS, variance and lag-1 AC for each simulation
#---------------------

# Filter time-series to have time-spacing dt2
df_traj_filt = df_traj.loc[::int(dt2/dt)]      

# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []
appended_pspec = []

# loop through realisation number
print('\nBegin EWS computation\n')
for i in range(numSims):
    # loop through variable
    for var in ['S','I']:
      ews_dic = ewstools.core.TimeSeries(df_traj_filt.loc[i+1][var], 
                                          transition=tbif[i])
      
      ews_dic.detrend(method='Lowess', span=span)                                
      ews_dic.compute_auto(rolling_window=rw, lag=1)                             
      ews_dic.compute_var(rolling_window=rw)                                     

      # The DataFrame of EWS
      z1 = ews_dic.state                                                         
      z2 = ews_dic.ews                                                           
      df_ews_temp = pd.concat([z1, z2], axis=1)                                 
    
      # Include a column in the DataFrames for realisation number and variable
      df_ews_temp['tsid'] = i+1
      df_ews_temp['Variable'] = var
    
      # Add DataFrames to list
      appended_ews.append(df_ews_temp)
        
    # Print status every realisation
    if np.remainder(i+1,1)==0:
      print('EWS for realisation '+str(i+1)+' complete')

# Concatenate EWS DataFrames
df_ews = pd.concat(appended_ews).reset_index().set_index(['tsid','Variable','Time'])

#Export individual resids files 
existing_simulation_number = 0
sim_index = (count-1)*numSims+existing_simulation_number

if not os.path.exists('../Training_Data/data_train_DemNoise/output_sims_batch{}'.format(count)):
    os.makedirs('../Training_Data/data_train_DemNoise/output_sims_batch{}'.format(count)) 

if not os.path.exists('../Training_Data/data_train_DemNoise/output_resids_batch{}'.format(count)):
    os.makedirs('../Training_Data/data_train_DemNoise/output_resids_batch{}'.format(count))

for i in np.arange(numSims)+1:
    Ind_tseries = df_ews.sort_index().loc[i,'I'].reset_index()[['Time','state']]
    tseries_rename = Ind_tseries.rename(columns={'Time': 'Time', 'state': 'I'})
    filepath='../Training_Data/data_train_DemNoise/output_sims_batch{}/tseries{}.csv'.format(count,sim_index+i)
    tseries_rename.to_csv(filepath,
                     index=False)    
    df_resids = df_ews.sort_index().loc[i,'I'].reset_index()[['Time','residuals']]
    filepath='../Training_Data/data_train_DemNoise/output_resids_batch{}/resids{}.csv'.format(count, sim_index+i)
    df_resids.to_csv(filepath,
                     index=False)

if not os.path.exists('../Training_Data/data_train_DemNoise/output_labels'):
    os.makedirs('../Training_Data/data_train_DemNoise/output_labels')
    
#label_indv
filepath = '../Training_Data/data_train_DemNoise/output_labels/label_batch{}.csv'.format(count)
label.to_csv(filepath,
            header=False, index=False) 

#store bifurcation_point
pd.DataFrame(tbif).to_csv('../Training_Data/data_train_DemNoise/output_labels/bifurcation_points_batch{}.csv'.format(count), header=False, index=False)

