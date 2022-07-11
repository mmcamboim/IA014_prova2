# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:55:33 2022

@author: mcamboim
"""
import numpy as np
import matplotlib.pyplot as plt

from model import dynamic_model
from generic_model import generic_dynamic_model
from intersection_method import intersection_method

plt.close('all')

plt.rcParams['axes.linewidth'] = 2.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

steps = 500
lines = 50

# Load Model -----------------------------------------------------------------
A,B,C,D = dynamic_model()

# Persistent excitation Signal ------------------------------------------------------
model = generic_dynamic_model(A,B,C,D)
#input_signal = np.random.normal(0,50,size=(steps,2))
input_signal = np.random.randint(0,2,size=(steps,2)) * 20
output_signal = np.zeros((steps,2))
for k in range(steps):
    exogenous_input = list(input_signal[k])
    model.iteration(exogenous_input)
    output_signal[k,:] = model.output.reshape(1,2)

# Intersection Method ----------------------------------------------------------
# Ver como deeterminar i e j de acordo com medições
# ajeitar o cod´gio
columns = steps + 1 - 2 * lines
intersec = intersection_method()
A_hat,B_hat,C_hat,D_hat = intersec.intersectionMethod(uk = input_signal, yk = output_signal, lines = lines, columns = columns)

# Models Comparison-----------------------------------------------------------
model_int = generic_dynamic_model(A_hat,B_hat,C_hat,D_hat)
model = generic_dynamic_model(A,B,C,D)

yk_model = np.zeros((steps,2))
yk_int = np.zeros((steps,2))
for k in range(steps):
    #exogenous_input = [np.sin(k/10),np.cos(k/10)]
    exogenous_input = [np.random.normal(0,10),np.random.normal(0,10)]
    model_int.iteration(exogenous_input)
    model.iteration(exogenous_input)
    yk_model[k,:] = model.output.reshape(1,2)
    yk_int[k,:] = model_int.output.reshape(1,2)

 
# Plots ----------------------------------------------------------------------
def plot_graphs(y11,y12,y21,y22,ylabel1,ylabel2,legend1,legend2,legend3,legend4,one_output):
    plt.figure(figsize=(6,6),dpi=150)
    plt.subplot(2,1,1)
    plt.plot(y11,c='b',lw=3)
    plt.plot(y12,c='r',lw=3)
    plt.legend([legend1,legend2])
    plt.xlabel('(a)')
    plt.ylabel(ylabel1)
    plt.grid(True,ls='dotted')
    
    plt.subplot(2,1,2)
    plt.plot(y21,c='b',lw=3)
    if(not one_output):
        plt.plot(y22,c='r',lw=3)
        plt.legend([legend3,legend4])
    plt.xlabel('Passo [k]\n(b)')
    plt.ylabel(ylabel2)
    plt.grid(True,ls='dotted')
    
    plt.tight_layout()
    

plot_graphs(input_signal[:,0],input_signal[:,1],output_signal[:,0],output_signal[:,1],
            'Entradas []','Saídas []','$u_1(k)$','$u_2(k)$','$y_1(k)$','$y_2(k)$',False)

plot_graphs(yk_model[:,0],yk_int[:,0],yk_model[:,0] - yk_int[:,0],[],
            'Saídas []','Erro []','$y_1(k): Modelo$','$y_1(k): Interseção$','','',True)

plot_graphs(yk_model[:,1],yk_int[:,1],yk_model[:,1] - yk_int[:,1],[],
            'Saídas []','Erro []','$y_2(k): Modelo$','$y_2(k): Interseção$','','',True)

plt.hist(yk_model[:,1] - yk_int[:,1],bins=20)

