"""
Created on August 2020

@author: Patrick van Meurs, Harmon Prayogi

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import time
import annf


# initialization
# tuneable parameters
n = 20               # the number of particle
T = 1                # the simulation time
d = 0.00001          # the cutoff distance before annihiliation
dt_0 = 0.001         # the initial time step 
k = 0                # first step
dt = np.array([dt_0])
tk = np.array([0])
f = np.zeros((1, n))
d_min = np.array([0])
d_plus = np.array([0])


random.seed(3)
x = np.array([[random.uniform(0,1) for i in range(n)]])
b = np.array([[random.choice([1, -1]) for i in range(n)]])
x.sort()


while tk[k] < T:
    # the distance between each particles in a matrix form
    r = annf.distance_each_particles(n, x[k])
    
    # the annihilation algorithm
    b_ann = annf.annihilation_rule(n, b[k], x[k], d)
    b = np.concatenate((b, np.array([b_ann])))
    
    # the calculation of the force term
    f_ann = annf.force(n, b[k+1], r)
    f = np.concatenate((f, np.array([f_ann])))
    
    # update the particle system, eliminate zero charge particle 
    # we use the copy of x, b, and f to the calculation below
    x_cal, b_cal, f_cal = annf.x_b_f_calculation(x[k], b[k], f[k+1])
    
    # calculate d^{-} and d^{+}
    xdmin, xdplus = annf.d_min_plus(x_cal, b_cal, d_min, d_plus)
    d_min = np.concatenate((d_min, np.array([xdmin.min()])))
    d_plus = np.concatenate((d_plus, np.array([xdplus.min()])))
    
    # stability condition
    h_1 = annf.first_stability_condition(x_cal, b_cal, f_cal)
    h_2 = annf.second_stability_condition(x_cal, b_cal, f_cal, d_plus[k+1])
    h_3 = annf.third_stability_condition(x_cal, b_cal, f_cal, d_min[k+1])
    h = 0.5 * np.concatenate((h_1, h_2, h_3, 2*np.array([dt_0])))
    
    # variable time step
    dt = np.concatenate((dt, np.array([h.min()])))
    tk = np.concatenate((tk, np.array([dt[k+1] + tk[k]])))
    
    # the explicit Euler method
    x_k = x[k] + dt[k+1] * f[k+1]
    x = np.concatenate((x, np.array([x_k])))
    
    # next time step
    k += 1
    
    
###################################### plotting ##############################################
##############################################################################################
## position plot linear t-axis
plt.figure(1,figsize=(10,8))

for i in range(int(n)):
    #plt.plot(tk[np.where(b.T[i] == 0)[0].tolist()],x[np.where(b.T[i] == 0)[0].tolist(),i],'g')
    plt.plot(tk[np.where(b.T[i] == 1)[0].tolist()],x[np.where(b.T[i] == 1)[0].tolist(),i],'r')
    plt.plot(tk[np.where(b.T[i] == -1)[0].tolist()],x[np.where(b.T[i] == -1)[0].tolist(),i],'b')

plt.xlabel('$t$', fontsize=18)
plt.ylabel('$x_i$', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([0,0.2])
#plt.ylim([-0.5,1.5])
plt.grid()

custom_lines = [Line2D([0], [0], color='b', lw=3),
                Line2D([0], [0], color='r', lw=3)]

plt.legend(custom_lines, ['b = -1', 'b = 1'], loc='upper right', prop={'size': 13})

#plt.savefig("figures/figure.pdf".format(37),bbox_inches='tight')

plt.show()


## position plot logarithmic t-axis
plt.figure(2,figsize=(10,8))

for i in range(int(n)):
    #plt.plot(tk[np.where(b.T[i] == 0)[0].tolist()],x[np.where(b.T[i] == 0)[0].tolist(),i],'g')
    plt.plot(tk[np.where(b.T[i] == 1)[0].tolist()],x[np.where(b.T[i] == 1)[0].tolist(),i],'r')
    plt.plot(tk[np.where(b.T[i] == -1)[0].tolist()],x[np.where(b.T[i] == -1)[0].tolist(),i],'b')

plt.xlabel('$t$', fontsize=18)
plt.ylabel('$x_i$', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale("log")
#plt.ylim([-0.5,1.5])
plt.grid()

custom_lines = [Line2D([0], [0], color='b', lw=3),
                Line2D([0], [0], color='r', lw=3)]

plt.legend(custom_lines, ['b = -1', 'b = 1'], loc='upper right', prop={'size': 13})

#plt.savefig("figures/figure-log.pdf".format(37),bbox_inches='tight')

plt.show()
