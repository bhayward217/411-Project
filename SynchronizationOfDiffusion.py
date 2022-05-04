#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 08:56:41 2019

@author: nolte

D. D. Nolte, Introduction to Modern Dynamics: Chaos, Networks, Space and Time, 2nd ed. (Oxford,2019)
"""

# https://www.python-course.eu/networkx.php
# https://networkx.github.io/documentation/stable/tutorial.html
# https://networkx.github.io/documentation/stable/reference/functions.html

import numpy as np
from scipy import integrate
from scipy import linalg
from matplotlib import pyplot as plt
import networkx as nx
from UserFunction import linfit
import time
from scipy import sparse

tstart = time.time()

plt.close('all')

Nfac = 90   # 25
N = 2     # 50


m0 = 1
mgas = 5
c0 = np.zeros(shape=(N,))
c0[0] = mgas;
g = 0.01
B = 0.1
dt = 1

omegaNatFinal = 5
mfinal = (mgas / N)+m0

omegaLargest = omegaNatFinal/np.sqrt((mgas+m0)/mfinal)

averageOmegaInit = np.average(omegaNatFinal/np.sqrt((c0+m0)/mfinal));

width = 1.4*omegaLargest


# model_case 1 = complete graph (Kuramoto transition)
# model_case 2 = Erdos-Renyi
model_case = 3#int(input('Input Model Case (1-2)'))
B = 0.01
if model_case == 1:
    facoef = 3 # Lengthens g axis???
    nodecouple = nx.complete_graph(N)
elif model_case == 2:
    facoef = 5
    nodecouple = nx.erdos_renyi_graph(N,0.1)
elif model_case == 3:
    # Makes a linearly coupled graph of n osccillators (in a loop)
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
    B = 0.09
dt = 1
laplacian = nx.laplacian_matrix(nodecouple, nodelist=None, weight='weight')
floqM = (sparse.eye(N)-laplacian*B*dt).toarray(order=None, out=None)
denseLap = laplacian.todense()
vals, vecs = np.linalg.eigh(denseLap)
v = np.zeros(N)
for i in range(N):
    Vtemp = vecs[:,i]
    v[i] = c0 @ Vtemp
    
#Gets the concentration (mass) each gas tank at time t0. Includes mass of tank Why t0 I'm not sure.

def gasDiffuse(y,t0):
    # transition = linalg.fractional_matrix_power(floqM,t0/dt)
    # cnew = transition @ c0 + m0
    concentration = np.zeros(N)
    for nodeloop in range(N):
         temp = 0;
         for eigloop in range(N): 
             temp = temp + vecs[nodeloop,eigloop]*v[eigloop]*np.exp(-vals[eigloop]*B*(t0));
            
         concentration[nodeloop] = temp;
    return concentration + m0;
    
# omega = np.linspace(-width/2, width/2, N)
# sto = np.std(omega)
 
# omega = np.zeros(shape=(N,))
# omegatemp = width*(np.random.rand(N)-1)
# omegatemp.sort()
# meanomega = np.mean(omegatemp)
# omega = omegatemp - meanomega
# sto = np.std(omega)

    
omega = np.zeros(shape=(N,))
omega[0] = width/2
omegatemp = omega
meanomega = np.mean(omegatemp)
omega = omegatemp - meanomega
sto = np.std(omega)



def coupleN(G, tc, oldOmega):

    # function: yd = flow_deriv(x_y)
    #nonLinOmega = omegaNatFinal/np.sqrt(masses/mfinal)'
    masses = gasDiffuse(0,tc);
    def flow_deriv(y,t0):
        #print(t0)
        yp = np.zeros(shape=(N,))
        #masses = (m0+c0)
        
        #print(masses)
        omega = omegaNatFinal/np.sqrt(masses/mfinal)
        
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

    mnomega = 1.0
    
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
    w = np.zeros(shape = (N,))
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

        
        #m[omloop] = np.median(mtmp)
        m[omloop] = np.mean(mtmp)
        
        w[omloop] = mnomega + m[omloop]
     
    omegout = m
    yout = y
    
    return omegout, yout
    # end of function: omegout, yout = coupleN(G)

lnk = np.zeros(shape = (N,), dtype=int)
for loop in range(N):
    nodecouple.nodes[loop]['element'] = omega[loop]
    nodecouple.nodes[loop]['link'] = list(nx.neighbors(nodecouple,loop))
    nodecouple.nodes[loop]['numlink'] = np.size(list(nx.neighbors(nodecouple,loop)))
    lnk[loop] = np.size(list(nx.neighbors(nodecouple,loop)))

avgdegree = np.mean(lnk)
mnomega = 1

# facval = np.zeros(shape = (Nfac,))
# yy = np.zeros(shape=(Nfac,N))
# xx = np.zeros(shape=(Nfac,))

# facval = np.zeros(shape = (Nfac,))
# yy = np.zeros(shape=(Nfac,N))
# xx = np.zeros(shape=(Nfac,))
# for facloop in range(Nfac):
#     print(facloop)

#     fac = facoef*(16*facloop/(Nfac))*(1/(N-1))*sto/mnomega
#     for nodeloop in range(N):
#         nodecouple.nodes[nodeloop]['coupling'] = np.zeros(shape=(lnk[nodeloop],))
#         for linkloop in range (lnk[nodeloop]):
#             nodecouple.nodes[nodeloop]['coupling'][linkloop] = fac

#     facval[facloop] = fac*avgdegree
    
#     omegout, yout = coupleN(nodecouple)                           # Here is the subfunction call for the flow

#     for omloop in range(N):
#         yy[facloop,omloop] = omegout[omloop]

#     xx[facloop] = facval[facloop]

# plt.figure(1)
# lines = plt.plot(xx,yy)
# plt.setp(lines, linewidth=0.5)
fac = g#facoef*(16/(Nfac))*(1/(N-1))*sto/mnomega
for nodeloop in range(N):
    nodecouple.nodes[nodeloop]['coupling'] = np.zeros(shape=(lnk[nodeloop],))
    for linkloop in range (lnk[nodeloop]):
        nodecouple.nodes[nodeloop]['coupling'][linkloop] = fac
tstop = 200
yy = np.zeros(shape=(tstop,N))
tt = np.zeros(shape=(tstop,))
omegaout = 0

oldOldOmega = averageOmegaInit
for i in range(tstop):
    omegaout, yout = coupleN(nodecouple,i,oldOldOmega)
    for omloop in range(N):
        yy[i,omloop] = omegaout[omloop]
    tt[i] = i;
    print(i)
    oldOldOmega = np.average(omegaout)
    print(oldOldOmega)
plt.figure(2)
ts = np.linspace(0,tstop-1,tstop);
cs = np.zeros(shape=(tstop,N));
for tunit in ts:
    cs[int(tunit)] = gasDiffuse(0,tunit)
lines2 = plt.plot(ts, cs)
plt.setp(lines2, linewidth = 0.5)
plt.title('Masses vs Time')
plt.xlabel('time')
plt.ylabel('total mass')

plt.figure()
plt.title('Frequencies vs Time for g={a}'.format(a = g))
lines = plt.plot(tt,yy)
plt.setp(lines, linewidth=0.5)
plt.xlabel('time')
plt.ylabel('frequency')

#plt.figure()
lines = plt.plot(tt,yy)
plt.setp(lines, linewidth=0.5)
#plt.xlim([24,47])

elapsed_time = time.time() - tstart
print('elapsed time = ',format(elapsed_time,'.2f'),'secs')


plt.figure()
diffFreq = np.zeros(shape=(tstop,N))
diffdiffFreq = np.zeros(shape=(tstop,N-1))
for i in range(tstop):
    for omloop in range(N):
        diffFreq[i,omloop] = yy[i,omloop]-omegaNatFinal/(np.sqrt(cs[i,omloop]/mfinal))
        if(omloop > 0):
            diffdiffFreq[i,omloop-1] = ((cs[i,omloop])/(cs[i,omloop-1]))**(omegaNatFinal/g)
            #diffdiffFreq[i,omloop-1] = g>omegaNatFinal/(np.sqrt(cs[i,omloop]/mfinal))-omegaNatFinal/(np.sqrt(cs[i,omloop-1]/mfinal))#yy[i,omloop] - yy[i, omloop-1]
    tt[i] = i;
    print(i)

plt.figure()
plt.title('Diff Freq Time for g={a}'.format(a = g))
#lines = plt.plot(tt,diffdiffFreq)#abs(diffdiffFreq[:,0]))#, color = "black")
plt.plot(tt,diffFreq)
#plt.setp(lines, linewidth=0.5)
#plt.yslim(-0.03,0.01)
plt.xlabel('time')
plt.ylabel('difffreq')

# plt.figure()
# plt.title('bleh for g={a}'.format(a = g))
# lines = plt.plot(tt,yy)
# lines2 = plt.plot(tt,diffdiffFreq)
# plt.setp(lines, linewidth=0.5)
# plt.ylim(-0.03,0.01)
# plt.xlabel('time')
# plt.ylabel('frequency difference')
# plt.show()