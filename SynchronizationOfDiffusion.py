"""
Created on Sat May 11 08:56:41 2019

@author: nolte (origional author)

Heavily modified by Ben Hayward
Wed May 4 20:14:23 2022

D. D. Nolte, Introduction to Modern Dynamics: Chaos, Networks, Space and Time, 2nd ed. (Oxford,2019)
"""

# https://www.python-course.eu/networkx.php
# https://networkx.github.io/documentation/stable/tutorial.html
# https://networkx.github.io/documentation/stable/reference/functions.html

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import networkx as nx
from UserFunction import linfit
import time
from scipy import sparse

tstart = time.time()

plt.close('all')

# Number of tanks
N = 2

# Mass of empty tank
m0 = 1

# Total mass of gas in tank network
mgas = 5

# Set up initial masses of gas in tanks
c0 = np.zeros(shape=(N,))
c0[0] = mgas;

# Coupling constant
g = 0.01

# Diffusion constant
B = 0.09

# Time step used in diffusion (not required elsewhere)
dt = 1

# Final natural frequency. Natural frequencies approach this frequency.
omegaNatFinal = 5

# Final mass of tanks, including empty tank mass. Assumes evenly distributed final state.
mfinal = (mgas / N)+m0

# Used to scale plots
omegaLargest = omegaNatFinal/np.sqrt((mgas+m0)/mfinal)
width = 1.4*omegaLargest


# model_case 1 = complete graph (Kuramoto transition)
# model_case 2 = Erdos-Renyi
# model_case 3 = Linear graph (what i used for my presentation)
model_case = 3      #int(input('Input Model Case (1-2)'))

# Diffusion constant
if model_case == 1:
    facoef = 3 # Lengthens g axis???
    nodecouple = nx.complete_graph(N)
elif model_case == 2:
    facoef = 5
    nodecouple = nx.erdos_renyi_graph(N,0.1)
elif model_case == 3:
    # Makes a linearly coupled graph of N osccillators (in a loop)
    a = np.zeros(shape = (N,N))
    for i in range(N):
        for j in range(N):
            if ((N + i - j) % N == 1) or ((N + j - i) % N == 1):
                a[i,j] = 1
            # if ((N + i - j) % N == 2) or ((N + j - i) % N == 2): #Small World
            #     a[i,j] = 1
    if(N>2):
        a[N-1,0] = 1
        a[0,N-1] = 1
    facoef = 3
    nodecouple = nx.from_numpy_matrix(a, parallel_edges=False, create_using=None)

# Using the network, finds the eigenvalues and eigenvectors of the graph laplacian
# once, rather than multiple times, to speed up the loops
laplacian = nx.laplacian_matrix(nodecouple, nodelist=None, weight='weight')
floqM = (sparse.eye(N)-laplacian*B*dt).toarray(order=None, out=None)
denseLap = laplacian.todense()
vals, vecs = np.linalg.eigh(denseLap)
v = np.zeros(N)
for i in range(N):
    Vtemp = vecs[:,i]
    v[i] = c0 @ Vtemp
    
# Gets the concentration (mass) each gas tank at time t0. Why t0 I'm not sure.
# Includes mass of tank
def gasDiffuse(y,t0):
    # Alternative, unused way to find the diffusion, using floquet matrix
    # transition = linalg.fractional_matrix_power(floqM,t0/dt)
    # cnew = transition @ c0 + m0
    
    #Method using eigenvectors
    concentration = np.zeros(N)
    for nodeloop in range(N):
         temp = 0;
         for eigloop in range(N): 
             temp = temp + vecs[nodeloop,eigloop]*v[eigloop]*np.exp(-vals[eigloop]*B*(t0));
         concentration[nodeloop] = temp;
    return concentration + m0;
 
# Natural frequencies of the oscillators   
omega = np.zeros(shape=(N,))

# At a time tc, calculates the frequency and position of the oscillators
def coupleN(G, tc):
    
    # Gets the masses at time tc
    masses = gasDiffuse(0,tc)
    
    def flow_deriv(y,t0):
        
        #This next coding line has to be inside function. I suspect it has 
        #something to do with how python manages garbage collection, but I 
        #don't know.
        
        #Natural frequencies
        omega = omegaNatFinal/np.sqrt(masses/mfinal)   
        #y partial derrivatives
        yp = np.zeros(shape=(N,))
        
        #Gets the derrivatives for each oscillator based on network
        #connectivity, coupling, and natural frequency
        for omloop  in range(N):
            temp = omega[omloop]
            linksz = G.nodes[omloop]['numlink']
            for cloop in range(linksz):
                cindex = G.nodes[omloop]['link'][cloop]
                g = G.nodes[omloop]['coupling'][cloop]
                temp = temp + g/N*np.sin(y[cindex]-y[omloop])
            
            yp[omloop] = temp
        
        yd = np.zeros(shape=(N,))
        for omloop in range(N):
            yd[omloop] = yp[omloop]
        return yd
    # end of function flow_deriv(x_y)
    
    for nodeloop in range(N):
        omega[nodeloop] = G.nodes[nodeloop]['element']
    
    x_y_z = omega    
    
    # Settle-down Solve for the trajectories
    tsettle = 100
    t = np.linspace(0, tsettle, tsettle)
    x_t = integrate.odeint(flow_deriv, x_y_z, t)
    x0 = x_t[tsettle-1,0:N]
    t = np.linspace(1,1000,1000)
    y = integrate.odeint(flow_deriv, x0, t)
    siztmp = np.shape(y)
    sy = siztmp[0]
        
    # Fit the frequency
    m = np.zeros(shape = (N,))
    mtmp = np.zeros(shape=(4,))
    btmp = np.zeros(shape=(4,))
    for omloop in range(N):
        
        if np.remainder(sy,4) == 0:
            mtmp[0],btmp[0] = linfit(t[0:sy//2],y[0:sy//2,omloop]);
            mtmp[1],btmp[1] = linfit(t[sy//2+1:sy],y[sy//2+1:sy,omloop]);
            mtmp[2],btmp[2] = linfit(t[sy//4+1:3*sy//4],y[sy//4+1:3*sy//4,omloop]);
            mtmp[3],btmp[3] = linfit(t,y[:,omloop]);
        else:
            sytmp = 4*np.floor(sy/4);
            mtmp[0],btmp[0] = linfit(t[0:sytmp//2],y[0:sytmp//2,omloop]);
            mtmp[1],btmp[1] = linfit(t[sytmp//2+1:sytmp],y[sytmp//2+1:sytmp,omloop]);
            mtmp[2],btmp[2] = linfit(t[sytmp//4+1:3*sytmp/4],y[sytmp//4+1:3*sytmp//4,omloop]);
            mtmp[3],btmp[3] = linfit(t[0:sytmp],y[0:sytmp,omloop]);

        
        m[omloop] = np.mean(mtmp)
     
    omegout = m
    yout = y
    
    return omegout, yout
# end of function: omegout, yout = coupleN(G)

# Not quite sure what this block does
lnk = np.zeros(shape = (N,), dtype=int)
for loop in range(N):
    nodecouple.nodes[loop]['element'] = omega[loop]
    nodecouple.nodes[loop]['link'] = list(nx.neighbors(nodecouple,loop))
    nodecouple.nodes[loop]['numlink'] = np.size(list(nx.neighbors(nodecouple,loop)))
    lnk[loop] = np.size(list(nx.neighbors(nodecouple,loop)))

avgdegree = np.mean(lnk)

# Sets the coupling of each link in network to g
fac = g
for nodeloop in range(N):
    nodecouple.nodes[nodeloop]['coupling'] = np.zeros(shape=(lnk[nodeloop],))
    for linkloop in range (lnk[nodeloop]):
        nodecouple.nodes[nodeloop]['coupling'][linkloop] = fac


# Gets the frequencies as a function of time. This is where the coupleN
# function is used
tstop = 200                    # Time until frequencies and masses are plotted
yy = np.zeros(shape=(tstop,N)) # Frequencies
tt = np.zeros(shape=(tstop,))  # Times
for i in range(tstop):
    omegaout, yout = coupleN(nodecouple,i) # coupleN used here
    for omloop in range(N):
        yy[i,omloop] = omegaout[omloop]
    tt[i] = i;

##############################################
#
#   Plotting Data
#
##############################################


# Recalculates the Masses vs time
ts = np.linspace(0,tstop-1,tstop);
cs = np.zeros(shape=(tstop,N));
for tunit in ts:
    cs[int(tunit)] = gasDiffuse(0,tunit)
    
# Plots the Masses vs time
lines2 = plt.plot(ts, cs)
plt.setp(lines2, linewidth = 0.5)
plt.title('Masses vs Time')
plt.xlabel('time')
plt.ylabel('total mass')
plt.figure()

# Plots the Frequencies vs time
plt.title('Frequencies vs Time for g={a}'.format(a = g))
lines = plt.plot(tt,yy)
plt.setp(lines, linewidth=0.5)
plt.xlabel('time')
plt.ylabel('frequency')
plt.figure()

# Prints the runtime
elapsed_time = time.time() - tstart
print('elapsed time = ',format(elapsed_time,'.2f'),'secs')
