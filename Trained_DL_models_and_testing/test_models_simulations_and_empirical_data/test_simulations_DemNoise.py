#Test simulations of SIR model with Demographic Noise
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
numSims = 20                           
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


#model parameters
Lambda = 100                                          
alpha = 1
mu = 1
S0 = 500                                   
I0 = 7

# bifurcation point
betabif = mu*(alpha+mu)/Lambda                   
print('betabif=', betabif)


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

print('beta_intercept=', beta_intercept)
print('beta_slope=', beta_slope)

list_traj_append = []                                                            
label_list = []
noise_intensity_S = []
noise_intensity_I = []
tbif = np.zeros(numSims) 
betas = []

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
    #print(tbif)
    
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
        I0 = random.uniform(0.1, 1)
                                                                   
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
        if I[i+1] < 0.1:
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
    
    betas.append(beta)
    noise_intensity_S.append(sigma_S)
    noise_intensity_I.append(sigma_I)

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

print ('tbif=', tbif)

plot_num = 11
var = 'I'

print(betas[plot_num-1])

fig, ax1 = plt.subplots() 
color = 'tab:blue' 
ax1.plot(df_ews.sort_index().loc[plot_num,var][['state','smoothing']])
ax1.set_xlabel('time')
ax1.set_ylabel('Infected Individuals', color = color)
plt.axvline(tbif[plot_num-1], color ='r', linestyle = 'dashed')
ax2 = ax1.twinx()                                           

color = 'tab:orange'
ax2.set_ylabel('beta', color=color)                         
ax2.plot(t, betas[plot_num-1], color=color) 
fig1, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,6), sharex=True)
df_ews.sort_index().loc[plot_num,var]['ac1'].plot(ax=axes[0],legend=True)
df_ews.sort_index().loc[plot_num,var]['variance'].plot(ax=axes[1],legend=True)


model = 'SIRdemN'

#Export labels
if not os.path.exists('../data/output_labels'):
    os.makedirs('../data/output_labels')
    
#label_indv = label.loc [0]
column_names = ['value']
label.columns = column_names
filepath = '../data/output_labels/label_SIR-{}.csv'.format(model)
label.to_csv(filepath,
            header=False, index=False) 

#Export tseries, residuals, and EWS dataframe 

if not os.path.exists('../data/sims'):
    os.makedirs('../data/sims') 

if not os.path.exists('../data/resids'):
    os.makedirs('../data/resids')

if not os.path.exists('../data/ews'):
    os.makedirs('../data/ews')

df_ews_null = df_ews.loc[1:10]

df_ews_forced = df_ews.loc[11:20]
df_ews_forced = df_ews_forced.reset_index()
df_ews_forced['tsid'] = df_ews_forced['tsid'].replace(list(range(11, 21)), list(range(1, 11)))


# Export EWS data
df_ews.to_csv('../data/ews/df_ews_{}.csv'.format(model))
df_ews_forced.to_csv('../data/ews/df_ews_forced_{}.csv'.format(model), index=False)
df_ews_null.to_csv('../data/ews/df_ews_null_{}.csv'.format(model))

beta_interceps_and_slopes = pd.DataFrame(list(zip(beta_slope, beta_intercept)), columns=['beta_slope', 'beta_intercept'])
beta_interceps_and_slopes.to_csv('../data/ews/slopes_and_intercepts_{}.csv'.format(model), index=False)

noise_intensity = pd.DataFrame(list(zip(noise_intensity_S, noise_intensity_I)), columns=['sigma_1', 'sigma_2'])
noise_intensity.to_csv('../data/ews/noise_intensity_{}.csv'.format(model), index=False)

forced = 1
null = 1 

for i in np.arange(numSims)+1:
    Ind_tseries = df_ews.sort_index().loc[i,'I'].reset_index()[['Time','state']]
    tseries_rename = Ind_tseries.rename(columns={'Time': 'Time', 'state': 'I'})
    
    if label['value'][i-1] == 1 : 
        filepath='../data/sims/tseries_{}_forced{}.csv'.format(model, forced)
        tseries_rename.to_csv(filepath,
                         index=False)    
        df_resids = df_ews.sort_index().loc[i,'I'].reset_index()[['Time','residuals']]
        filepath='../data/resids/resids_{}_forced{}.csv'.format(model, forced)
        df_resids.to_csv(filepath,
                         index=False)
        forced = forced+1
    else:
        filepath='../data/sims/tseries_{}_null{}.csv'.format(model, null)
        tseries_rename.to_csv(filepath,
                         index=False)    
        df_resids = df_ews.sort_index().loc[i,'I'].reset_index()[['Time','residuals']]
        filepath='../data/resids/resids_{}_null{}.csv'.format(model, null)
        df_resids.to_csv(filepath,
                         index=False)
        null = null+1


