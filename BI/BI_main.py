#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:59:39 2022

"""

import numpy as np
import os
import scipy.io

import matplotlib.pyplot as plt
current = os.getcwd()
import pandas as pd
import datetime
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker

import matplotlib.gridspec as gridspec
import statistics
import seaborn as sns
from sklearn.metrics import mean_squared_error


color_mu = 'tab:blue'
color_k = 'tab:red'
color_b = 'tab:green'


ColorS = [0.5, 0.00, 0.0]
ColorE = [0.8, 0.00, 0.0]
ColorI = [1.0, 0.65, 0.0]
ColorR = [0.0, 0.00, 0.7]


filename = os.path.basename(__file__)[:-3]
wd = os.path.abspath(os.getcwd())

def makeDIR(phad):
    if not os.path.exists(phad):
        os.makedirs(phad)
        
        
path2saveResults = 'Results/'+filename
makeDIR(path2saveResults)


trainmodel = True


import pymc3 as pm
import theano
import theano.tensor as tt



def Ogden(mu,alpha,strain):
    # Initialization
    lam_1 = strain

    stress = ( mu * (lam_1**alpha - (lam_1)**(-alpha/2) ) )/lam_1

    return stress








covid_world = np.loadtxt('covid_world.dat')

days = np.arange(0,covid_world.shape[0])

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

covid_world_smooth = movingaverage(covid_world[:,1],7)
years = days/365


plt.figure(figsize=(1100/72,400/72))
plt.plot(years,covid_world[:,1],color='blue',label='daily new cases')
plt.plot(years,covid_world_smooth,linewidth=2.0,color='red', label='smoothed daily new cases')
plt.xlabel('time [years]')
plt.ylabel('daily new cases worldwide')
plt.legend(loc='upper left',fontsize=14)


# In[3]:


# plot window of interest
d1 = 345 
d2 = 695

plt.figure(figsize=(1100/72,400/72))
plt.plot((days[d1:d2]-d1)/365,covid_world[d1:d2,1]/1e06,color='blue',label='daily new cases')
plt.plot((days[d1:d2]-d1)/365,covid_world_smooth[d1:d2]/1e06,linewidth=2.0,color='red', label='smoothed daily new cases')
plt.xlabel('time [years]')
plt.ylabel('daily new cases worldwide')
plt.legend(loc='upper left',fontsize=14)


# Test around for some good start values for the model parameters

# In[4]:


# analytical solution underdamped oscillator: try to find approximate match for model parameters
def oscillator(d, w0, b0, A_mod, t):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
#     A = A_mod * 1/(2*np.cos(phi))
    A = A_mod
    cosine = np.cos(phi+w*t)
    # sine = np.sin(phi+w*t)
    exp = np.exp(-d*t)
    y  = exp*2*A*cosine + b0
    return y




# In[5]:


# model parameters
d, w0, b0, A_mod = 1.1, 20.2, 0.56, 0.15

# get the analytical solution over the full domain
t_ana = np.linspace(0,1,500)
x_ana = oscillator(d, w0, b0, A_mod, t_ana)

plt.figure(figsize=(1100/72,400/72))
plt.plot((days[d1:d2]-d1)/365,covid_world[d1:d2,1]/1e06,color='blue',label='daily new cases')
plt.plot((days[d1:d2]-d1)/365,covid_world_smooth[d1:d2]/1e06,linewidth=2.0,color='red', label='smoothed daily new cases')
plt.plot(t_ana, x_ana, color='black', linewidth=2.0, label='analytical solution oscillator')
plt.xlabel('time [years]')
plt.ylabel('daily new cases worldwide')
plt.legend(loc='upper right',fontsize=14)


# Now, for the ODE: $k=m\omega_0^2$ and $\mu = 2m\delta$

# In[6]:


#### create training data

t_covid = days[d1:d2]-d1   # time array: days
y_covid = t_covid/365      # time array: years
x_covid = covid_world_smooth[d1:d2]/1e06  # normalize COVID numbers per 10^6 people 

# pick training data
t_data = y_covid[7:200:7]  # weekly data points work better than daily
x_data = x_covid[7:200:7]

t_weekly = y_covid
x_weekly = x_covid



# collocation points for enforcing ODE, minimizing residual
t_physics = y_covid[0::7]




def BImodel(cutoff_idx):

    # Masked array used to train and predict the model
    y_obs = np.ma.MaskedArray(x_weekly, t_weekly>cutoff_idx)
    
    
    #%% Definition
    
    with pm.Model() as model:
    
        
        log_mu = pm.Normal('log_mu', mu=np.log(2.2), sigma=0.5)
        log_k = pm.Normal('log_k', mu=np.log(350.0), sigma=0.5)
        log_b = pm.Normal('log_b', mu=np.log(0.56), sigma=0.5)
        log_A = pm.Normal('log_A', mu=np.log(0.5), sigma=0.5)
        
        mu_par = pm.Deterministic('mu_par',  tt.exp(log_mu) ) # in ellen's notation "c"
        k_par = pm.Deterministic('k_par',  tt.exp(log_k) )
        
        w0_tt = pm.Deterministic('w0',  tt.sqrt(tt.exp(log_k)))
        d_tt =  pm.Deterministic('d_tt',  tt.exp(log_mu)/2.0)    
    
        b0_tt = pm.Deterministic('b0_tt',  tt.exp(log_b))
        A_mod_tt = pm.Deterministic('Amod_tt',  tt.exp(log_A))
        
        pm.Potential('ordered', tt.switch(w0_tt-d_tt < 0, -np.inf, 0))
     
        phi_par = pm.Deterministic('phi_par',  tt.arctan(-d_tt/ (tt.sqrt(tt.pow(w0_tt,2)- tt.pow(d_tt,2))) ) )
        
        # analyticalmodel
        w = tt.sqrt(tt.pow(w0_tt,2)- tt.pow(d_tt,2))
        phi = tt.arctan(-d_tt/w)
    
        cosine = tt.cos(phi+ (w * t_weekly))
        exp = tt.exp(-d_tt * t_weekly)
        ODE_model  = exp *2* A_mod_tt *cosine + b0_tt
        
        
        # sd = pm.HalfNormal('sd', 2, shape=2)
        
        σ_obs = pm.HalfCauchy('σ_obs', beta=1)
        
        pm.Normal('obs_', mu=ODE_model, sigma=σ_obs, observed=y_obs)
        #-------------------------------------------------------------------------- #
        #run model, pm trains and predicts when calling this
        #-------------------------------------------------------------------------- #
        
        if trainmodel:
            trace = pm.sample(draws=4000, tune=1000, chains=4, cores=2)
        else:
            trace = pm.load_trace(directory='traces/'+filename+'/.pymc_6.trace')
          
    pm.traceplot(trace);
    
    
    if trainmodel:
        pm.save_trace(trace,directory='traces/'+filename+'/.pymc_6.trace',overwrite=True)
      
    
    with model:
        ppc = pm.sample_posterior_predictive(trace, 1500)
        
        
    return ppc, trace
        



def plotting(N, ppc,y_covid,x_covid,t_weekly,days,d1,d2):
    model = np.median(ppc['obs_'],axis=0)
    lower = (np.percentile(ppc['obs_'], q=2.5, axis=0))
    upper = (np.percentile(ppc['obs_'], q=97.5, axis=0))
        
    
    plt.figure(figsize=(1100/72,400/72))
    
    y_points = np.ma.MaskedArray(x_covid, y_covid>cutoff_idx)
    
    plt.scatter(y_covid, y_points, s=300,edgecolors=(0.0, 0.0, 0.7),marker='.',facecolors='none', lw=1.0,zorder=1, alpha=1.0, label=r'train data') 
    # ax1.plot(Strain_mat[::inc,j], Stress_mat[::inc,j],color='k', lw=0.5, zorder=3, alpha =0.5, label='')
    # ax1.text(0.05,0.9,r'$R^2$: '+f"{R2_sample:.2f}",transform=ax1.transAxes,fontsize='large', horizontalalignment='left')
    plt.fill_between(t_weekly, lower, upper, zorder=1, alpha=0.10, color=ColorR)
    plt.plot((days[d1:d2]-d1)/365,covid_world[d1:d2,1]/1e06,color=ColorI, lw=1.0, alpha=1.0,label='daily new cases')
    
    plt.plot(t_weekly, model, color=(0.7, 0.0, 0.0),linewidth=3.0,linestyle="--", zorder=3, alpha=1.0, label=r'Fit new confirmed cases (with 95% CI)')
    plt.plot((days[d1:d2]-d1)/365,covid_world_smooth[d1:d2]/1e06,linewidth=3.0,zorder=2,color=ColorI, label='smoothed daily new cases')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    
    plt.xlabel("time [years]",fontsize=22)
    plt.ylabel("cases [M]",fontsize=18)
    plt.ylim([0,1])
    # plt.legend(loc='upper left',fontsize=14,ncol=2, fancybox=True, framealpha=0.)
    plt.savefig('ODE-BI_model.pdf')
    
    
    plt.savefig(path2saveResults+'/Oscillator_Bay_ana_'+str(N)+'_v1.pdf')
    plt.show()
    plt.close()
    
    
    



# Run model

N =225
cutoff_idx = N/350
ppc, trace = BImodel(cutoff_idx)
plotting(N, ppc,y_covid,x_covid,t_weekly,days,d1,d2)






#%%
# Posteriors

def PlotHist(ax, prior, post, var, color, limY, limX, limX0):
    # ax.hist(prior, bins=30, density=True, zorder=5, label='prior of '+var, color=color, alpha=0.3)
    ax.hist(post, bins=num_bins, density=True, label='posterior of '+var, color=color, alpha=0.7)
    mean = np.round(np.mean(post),3)
    std = np.round(np.std(post),2)
    ax.set_ylim([0,limY])
    ax.set_xlim([limX0,limX])
    # plt.title('mean of '+var+': '+str(mean),fontsize=24, color=color)
    plt.title(var+' = '+str(mean)+'$\pm$'+str(std),fontsize=24, color=color)
    legend = plt.legend(loc='upper right',fontsize=24,ncol=1, fancybox=True, framealpha=0.)
    plt.setp(legend.get_texts(), color=color)
    



N = 30000
num_bins = 30
color_A = 'k'


fig = plt.figure(figsize=(1200/72,800/72))
gs = fig.add_gridspec(2, 2)

mu = 0.55
sigma = 0.5
prior = np.exp(sigma * np.random.randn(N) + np.log(mu))
post = trace['Amod_tt']
var = '$A_0$'
ax1 = plt.subplot(gs[0, 0])

PlotHist(ax1, prior, post, var, color_A, 160, 0.14, 0.08)
    

mu = 2.2
sigma = 0.5
prior = np.exp(sigma * np.random.randn(N) + np.log(mu))
post = trace['mu_par']
var = '$c$'
ax2 = plt.subplot(gs[0, 1])


PlotHist(ax2, prior, post, var, color_mu, 3.8, 1.6, 0.6)



mu = 350
sigma = 0.5
prior = np.exp(sigma * np.random.randn(N) + np.log(mu))
post = trace['k_par']
var = '$k$'
ax3 = plt.subplot(gs[1, 0])


PlotHist(ax3, prior, post, var, color_k, 0.27, 420, 395)




mu = 0.56
sigma = 0.5
prior = np.exp(sigma * np.random.randn(N) + np.log(mu))
post = trace['b0_tt']
var = '$x_0$'
ax4 = plt.subplot(gs[1, 1])


PlotHist(ax4, prior, post, var, color_b, 220, 0.56, 0.53)

plt.savefig(path2saveResults+'/Oscillator_BI_Para_v2.pdf')







