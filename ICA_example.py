# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:01:03 2024

@author: BenPa


An example function designed to generate 2 mixed signals and separate them again using ICA
"""

import numpy as np 
import matplotlib.pyplot as plt



low, high =-1, 1
size=1000
# Generate beta data
uniform1 = np.random.uniform(low, high, size)
uniform2 = np.random.uniform(low, high, size)
# Plot the distribution
#plt.figure(figsize=[10,10])
plt.scatter(uniform1,uniform2)

plt.title("Joint distribution of two independent components with uniform densities")
plt.show()

scale = 1.0  # this is 1 / lambda
size = 1000

# Generate exponential data
exp_data = np.random.exponential(scale, size)


#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import matplotlib.gridspec as gridspec
# Step 1: Generate two independent uniform distributions
n_samples = 1000
s1 = np.random.uniform(-1, 1, n_samples)  # Uniform distribution 1
s2 = np.random.uniform(-1, 1, n_samples)  # Uniform distribution 2

# Stack them into a 2D array to represent the sources
S = np.c_[s1, s2].T

# Step 2: Mix the distributions with a random mixing matrix
A = np.array([[1, 0.5], [0.5, 1]])  # Mixing matrix
X = A@S # Mixed signals

# Step 3: Apply FastICA to recover the sources
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X.T)  # Recovered signals (estimated sources)

w= ica.components_


# Plotting the results
fig=plt.figure(figsize=(18, 6))

#outer grid creation
outer_grid = gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.3)

X_labels= ["Source 1","Mixed Signal 1","Recovered Component 1"]
Y_labels=["Source 2","Mixed Signal 2", "Recovered Component 2"]
scattersx=[S[0,:],X[0,:],S_[:,0]]
scattersy=[S[1,:],X[1,:],S_[:,1]]
colors=["r","b","g"]
for i in range(3):
    # Create an inner grid within each cell of the outer grid
    inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_grid[i], hspace=0.05, wspace=0.05)
    
    # Scatter plot (center)
    ax_main = fig.add_subplot(inner_grid[1:4, 0:3])
    ax_main.scatter(scattersx[i], scattersy[i], color=colors[i],alpha=0.6)
    ax_main.set_xlabel(X_labels[i])
    ax_main.set_ylabel(Y_labels[i])
    
    # X histogram (top)
    ax_histx = fig.add_subplot(inner_grid[0, 0:3], sharex=ax_main)
    ax_histx.hist(scattersx[i], bins=30, color='gray', alpha=0.7)
    ax_histx.axis('off')
    
    # Y histogram (right)
    ax_histy = fig.add_subplot(inner_grid[1:4, 3], sharey=ax_main)
    ax_histy.hist(scattersy[i], bins=30, color='gray', alpha=0.7, orientation='horizontal')
    ax_histy.axis('off')

plt.show()



#%%


# Plot original sources
fig.subplot(1, 3, 1)
inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_grid[0], hspace=0.05, wspace=0.05)
ax_main= fig.add_subplot(inner_grid[1:4, 0:3])
ax_main.scatter(S[0, :], S[1,:], alpha=0.5, color='green')
ax_main.set_title("Original Signals (Independent Sources)")
ax_main.set_xlabel()
ax_main.set_ylabel()

# Plot mixed signals
fig.subplot(1,3, 2)
inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_grid[1], hspace=0.05, wspace=0.05)
ax_main= fig.add_subplot(inner_grid[1:4, 0:3])
ax_main.scatter(X [0,:], X[1, :], alpha=0.5, color='red')
ax_main.set_title("Mixed Signals")
ax_main.set_xlabel("Mixed Signal 1")
ax_main.set_ylabel("Mixed Signal 2")

# Plot recovered signals (independent components)
fig.subplot(1,3, 3)
ax_main.scatter(S_[ :,0], S_[:,1], alpha=0.5, color='blue')
ax_main.set_title("Recovered Signals (Independent Components)")
ax_main.set_xlabel("Recovered Component 1")
ax_main.set_ylabel("Recovered Component 2")

plt.tight_layout()
plt.show()
