#!/usr/bin/env python
# coding: utf-8

# # 1D harmonic oscillator PINN for COVID
# 

# ## 1D oscillator equations
# 
# 1D damped harmonic oscillator (from https://beltoforion.de/en/harmonic_oscillator/):
# $$
# m \dfrac{d^2 x}{d t^2} + \mu \dfrac{d x}{d t} + kx = 0~,
# $$
# with initial conditions
# $$
# x(0) = 1~~,~~\dfrac{d x}{d t} = 0~.
# $$
# For the under-damped state, i.e. when 
# $$
# \delta < \omega_0~,~~~~~\mathrm{with}~~\delta = \dfrac{\mu}{2m}~,~\omega_0 = \sqrt{\dfrac{k}{m}}~.
# $$
# we can find the following exact solution:
# $$
# x(t) = e^{-\delta t}(2 A \cos(\phi + \omega t))~,~~~~~\mathrm{with}~~\omega=\sqrt{\omega_0^2 - \delta^2}~.
# $$
# 

# In[1]:


# import necessary python packages

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)


# ### 1. Load COVID data
#  

# In[2]:


covid_world = np.loadtxt('covid_world.dat')

days = np.arange(0,covid_world.shape[0])

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

covid_world_smooth = movingaverage(covid_world[:,1],7)
years = days/365




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
    sine = np.sin(phi+w*t)
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

# collocation points for enforcing ODE, minimizing residual
t_physics = y_covid[0::7]


# In[7]:


# convert arrays to tf tensors
t_data_tf = tf.convert_to_tensor(t_data, dtype=DTYPE)
x_data_tf = tf.convert_to_tensor(x_data, dtype=DTYPE)
t_physics_tf = tf.convert_to_tensor(t_physics, dtype=DTYPE)

T_data = tf.reshape(t_data_tf[:], shape=(t_data.shape[0],1))
X_data = tf.reshape(x_data_tf[:], shape=(x_data.shape[0],1))
T_r = tf.reshape(t_physics_tf[:], shape=(t_physics.shape[0],1))


# In[8]:


# weighting factor for losses : alpha*loss_data + (1-alpha)*loss_r
alpha = 0.9995


# In[9]:


# Define model architecture
class PINNIdentificationNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, 
            output_dim=1,
            num_hidden_layers=2, 
            num_neurons_per_layer=32,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        
        # Define NN architecture
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
        # Initialize variable for mu, k: if trainable=True, will be trained along NN
        self.mu = tf.Variable(2.2, trainable=True, dtype=DTYPE)
        self.mu_list = []
        
        self.k = tf.Variable(408.0, trainable=True, dtype=DTYPE)
        self.k_list = []
        
        self.m = tf.Variable(1.0, trainable=False, dtype=DTYPE)
        self.m_list = []
        
        # x-offset around which trajectory oscillates 
        self.b = tf.Variable(0.56, trainable=True, dtype=DTYPE)
        self.b_list = []
        
        
        
    def call(self, T):
        """Forward-pass through neural network."""
        Z = self.hidden[0](T)
        for i in range(1,self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)


# In[10]:


class PINNSolver_ID():
    def __init__(self, model, T_r):
        self.model = model
        
        # Store collocation points
        self.t = T_r[:,0:1]
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
    
    def get_r(self):
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t during this GradientTape
            tape.watch(self.t)
            
            # Compute current values u(t,x)
            u = self.model(self.t[:,0:1])
            
            u_t = tape.gradient(u, self.t)
            
        u_tt = tape.gradient(u_t, self.t)
        
        del tape
        
        return self.fun_r(u, u_t, u_tt)
    
    def loss_fn(self, T, u):
        
        # Compute phi_r: loss coming from residual
        r = self.get_r()
        phi_r = (1-alpha)*tf.reduce_mean(tf.square(r))
        
        # Initialize loss
        loss = phi_r

        # Add loss coming from difference to training data
        for i in range(len(T)):
            u_pred = self.model(T[i:i+1,0:1])
            loss += alpha*tf.reduce_mean(tf.square(u[i] - u_pred))
        
        return loss
    
    def get_grad(self, T, u):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(T, u)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    
    def fun_r(self, u, u_t, u_tt):
        return self.model.m*u_tt + self.model.mu*u_t + self.model.k*(u-self.model.b) # b is vertical offset from t-axis
    
    def solve_with_TFoptimizer(self, optimizer, T, u, N=10000):
        """This method performs a gradient descent type optimization."""
        
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad(T, u)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        for i in range(N):
            
            loss = train_step()
            
            self.current_loss = loss.numpy()
            self.callback()
            
    def callback(self, tr=None):
        mu = self.model.mu.numpy()
        self.model.mu_list.append(mu)
        
        k = self.model.k.numpy()
        self.model.k_list.append(k)
        
        b = self.model.b.numpy()
        self.model.b_list.append(b)
        
        if self.iter % 100 == 0:
            print('It {:05d}: loss = {:10.8e} mu = {:10.8e} k = {:10.8e} b = {:10.8e}'.format(self.iter, self.current_loss, mu, k, b))
        self.hist.append(self.current_loss)
        self.iter+=1
        
    
    def plot_solution(self, **kwargs):
        
        n=d2-d1
        t_pred = np.reshape(np.linspace(0,n,n),(n,1))/365
        x_pred = self.model(t_pred)

        # plot prediction
        fig, ax2 = plt.subplots(figsize=(700/72,500/72))
        ax2.set_title('PINN COVID')
        ax2.scatter(t_data,x_data,s=300,color="tab:orange", alpha=1.0,marker='.') #observed data points 
        ax2.plot((days[d1:d2]-d1)/365,covid_world[d1:d2,1]/1e06,color='blue')
        ax2.plot((days[d1:d2]-d1)/365,covid_world_smooth[d1:d2]/1e06,linewidth=2.0,color='red')
        ax2.plot(t_pred,x_pred,color="black",linewidth=3.0,linestyle="--")
        ax2.legend(('training data','daily cases','daily cases smooth','model prediction'), loc='upper right',fontsize=14)
        ax2.set_xlabel('time [years]',fontsize=14)
        ax2.set_ylabel('daily new cases [per 10^6]]',fontsize=14)
        
        plt.savefig('Cov_PINN.pdf',bbox_inches='tight')
        
        return ax2
        
    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(700/72,500/72))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist,'k-')
        ax.set_xlabel('$n_{epoch}$',fontsize=18)
        ax.set_ylabel('$\\phi^{n_{epoch}}$',fontsize=18)
        return ax
    
    def plot_loss_and_param(self, axs=None):

        color_mu = 'tab:blue'
        color_k = 'tab:red'
        color_b = 'tab:green'

        fig = plt.figure(figsize=(1200/72,800/72))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = plt.subplot(gs[0, 0])
        ax1 = self.plot_loss_history(ax1)
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(range(len(self.hist)), self.model.mu_list,'-',color=color_mu)
        ax2.set_ylabel('$\\mu^{n_{epoch}}$', color=color_mu, fontsize=18)
        ax2.set_xlabel('$n_{epoch}$',fontsize=18)
        
        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(range(len(self.hist)), self.model.k_list,'-',color=color_k)
        ax3.set_ylabel('$k^{n_{epoch}}$', color=color_k, fontsize=18)
        ax3.set_xlabel('$n_{epoch}$',fontsize=18)
        
        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(range(len(self.hist)), self.model.b_list,'-',color=color_b)
        ax4.set_ylabel('$b^{n_{epoch}}$', color=color_b, fontsize=18)
        ax4.set_xlabel('$n_{epoch}$',fontsize=18)

        return (ax1,ax2,ax3,ax4)


# In[11]:


# Initialize model
model = PINNIdentificationNet()
model.build(input_shape=(None,1))

# Initialize PINN solver
solver = PINNSolver_ID(model, T_r)

# Start timer
t0 = time()


#lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([3000,6000],[1e-2,1e-3,5e-4])
lr = 5e-3
optim = tf.keras.optimizers.Adam(learning_rate=lr)
solver.solve_with_TFoptimizer(optim, T_data, X_data, N=17000)


# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))


# In[12]:


ax = solver.plot_solution();
axs = solver.plot_loss_and_param();


# In[ ]:




