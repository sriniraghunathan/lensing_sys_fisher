import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np, scipy as sc, sys, argparse, os

import json
import pandas as pd
import re
import tools

#exec(open('plot_results.py').read())

parser = argparse.ArgumentParser(description='')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt')

args = parser.parse_args()
args_keys = args.__dict__

for kargs in args_keys:
    param_value = args_keys[kargs]
    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

rms_map_T_list = np.arange(1,11,1)


dl = 5
Lsdl = dl
camborself = "self"
binsize = 10


with open("results/F_mat_CDMp_prior_%s_bin%s_dl%s_%s_n%s.json"%("delensed_scalar", binsize, Lsdl, camborself, rms_map_T_list[5])) as infile:
    F_matdelen = json.load(infile)
with open("results/F_mat_CDMp_prior_%s_bin%s_dl%s_%s_n%s.json"%("total", binsize, Lsdl, camborself, rms_map_T_list[5])) as infile:
    F_matlen = json.load(infile)
with open("results/F_mat_CDMp_prior_%s_bin%s_dl%s_%s_n%s.json"%("unlensed_total", 5, 5, "self", rms_map_T_list[5])) as infile:
    F_matunlen = json.load(infile)


logline = '\tget/read the parameter file'; tools.write_log(logline)
param_dict = tools.get_ini_param_dict(paramfile)

param_list = F_matlen['parms']

para_value = {}
for item in param_list:
    para_value[item] = param_dict[item]

Fmatdelen = np.asarray(F_matdelen['Fmat'])
cov_matdelen = np.asarray(F_matdelen['cov_mat'])
Fmatlen = np.asarray(F_matlen['Fmat'])
cov_matlen = np.asarray(F_matlen['cov_mat'])
Fmatunlen = np.asarray(F_matunlen['Fmat'])
cov_matunlen = np.asarray(F_matunlen['cov_mat'])
Fmatlen = Fmatdelen
cov_matlen = cov_matdelen
pnum = len(param_list)

print(cov_matlen)

#plt.figure(figsize = (12,12))
#fig, axes = plt.subplots(figsize=(10, 10), sharex = 'col',  sharey = 'row', ncols=pnum, nrows=pnum)
fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum, nrows=pnum)
#fig = plt.figure()
#gs = fig.add_gridspec(pnum, pnum, hspace=0, wspace=0)
#axes = plt.subplots()
#gs = fig.add_gridspec(hspace=0, wspace=0)

for i, p1 in enumerate(param_list):
    for j, p2 in enumerate(param_list):
        if i<j:
            axes[i,j].axis('off')
            #for j, p2 in enumerate(param_list[:i+1]):        
        elif i==j:
            sigmai = cov_matlen[i,i]**0.5
            valuei = para_value[p1]
            xst = valuei - sigmai*4
            xed = valuei + sigmai*4
            x = np.linspace(xst, xed, 40)            
            coe = 1.0 / (2 * np.pi)**0.5 / sigmai
            Z = np.exp (-0.5 * ((x-valuei)/sigmai)**2)
            axes[i,i].plot(x, Z)
            print('i = ',i)
            #axes[i,j].xlim
        else:
            sigmai = cov_matlen[i,i]**0.5
            sigmaj = cov_matlen[j,j]**0.5
            valuei = para_value[p1]
            valuej = para_value[p2]
            xst = valuej - sigmaj*5
            xed = valuej + sigmaj*5
            yst = valuei - sigmai*5
            yed = valuei + sigmai*5
            x = np.linspace(xst, xed, 40)
            y = np.linspace(yst, yed, 40)
            X,Y = np.meshgrid(x,y)
            coe = 1.0 / ((2 * np.pi)**2 * np.linalg.det(cov_matlen))**0.5
            Z = np.e ** (-0.5 * (Fmatlen[j,j]*(X-valuej)**2 + (Fmatlen[j,i] + Fmatlen[i,j])*(X-valuej)*(Y-valuei) + Fmatlen[i,i]*(Y-valuei)**2))
            z1 = np.exp(-0.5 * (Fmatlen[j,j]*(sigmaj)**2 + (Fmatlen[j,i] + Fmatlen[i,j])*(sigmaj)*(sigmai) + Fmatlen[i,i]*(sigmai)**2))
            z3 = np.exp(-0.5 * (Fmatlen[j,j]*(3*sigmaj)**2 + (Fmatlen[j,i] + Fmatlen[i,j])*(3*sigmaj)*(3*sigmai) + Fmatlen[i,i]*(3*sigmai)**2))
            #print('z3, z1',z3, z1)
            cs = axes[i,j].contourf(X,Y,Z,[z3, z1, 1], colors=['yellowgreen', 'olive', 'honeydew'])
            cs.cmap.set_over('blue')
            axes[i,j].set_ylim([yst, yed])
            axes[i,j].set_xlim([xst, xed])
            print('i, j', i,j ,p1, p2)
            print('xst, yst', xst, yst)

            if j == 0:
                axes[i,j].set_ylabel(p1)
            if i == pnum-1:
                axes[i,j].set_xlabel(p2)
plt.subplots_adjust(wspace=0, hspace=0)
#plt.tight_layout()
plt.savefig('compare_CDMp_prior_cov_noise6_phi_dl%s_%s_delen.png'%(dl, camborself))

'''

for i, p1 in enumerate(param_list):
    for j, p2 in enumerate(param_list):
        if i<j:
            axes[i,j].axis('off')
            #for j, p2 in enumerate(param_list[:i+1]):        
        elif i==j:
            sigmailen = cov_matlen[i,i]**0.5
            sigmaiunlen = cov_matunlen[i,i]**0.5
            sigmaidelen = cov_matdelen[i,i]**0.5
            valuei = para_value[p1]
            xst = valuei - sigmailen*4
            xed = valuei + sigmailen*4
            x = np.linspace(xst, xed, 40)            
            coe = 1.0 / (2 * np.pi)**0.5 / sigmai
            Z1 = np.exp (-0.5 * ((x-valuei)/sigmailen)**2)
            Z2 = np.exp (-0.5 * ((x-valuei)/sigmaidelen)**2)
            Z3 = np.exp (-0.5 * ((x-valuei)/sigmaiunlen)**2)
            axes[i,i].plot(x, Z1)
            axes[i,i].plot(x, Z2)
            axes[i,i].plot(x, Z3)
            print('i = ',i)
            #axes[i,j].xlim
        else:
            sigmailen = cov_matlen[i,i]**0.5
            sigmaiunlen = cov_matunlen[i,i]**0.5
            sigmaidelen = cov_matdelen[i,i]**0.5
            valuei = para_value[p1]
            valuej = para_value[p2]
            xst = valuej - sigmajlen*5
            xed = valuej + sigmajlen*5
            yst = valuei - sigmailen*5
            yed = valuei + sigmailen*5
            x = np.linspace(xst, xed, 40)
            y = np.linspace(yst, yed, 40)
            X,Y = np.meshgrid(x,y)
            coe = 1.0 / ((2 * np.pi)**2 * np.linalg.det(cov_matlen))**0.5
            Z = np.e ** (-0.5 * (Fmatlen[j,j]*(X-valuej)**2 + (Fmatlen[j,i] + Fmatlen[i,j])*(X-valuej)*(Y-valuei) + Fmatlen[i,i]*(Y-valuei)**2))
            z1 = np.exp(-0.5 * (Fmatlen[j,j]*(sigmaj)**2 + (Fmatlen[j,i] + Fmatlen[i,j])*(sigmaj)*(sigmai) + Fmatlen[i,i]*(sigmai)**2))
            z3 = np.exp(-0.5 * (Fmatlen[j,j]*(3*sigmaj)**2 + (Fmatlen[j,i] + Fmatlen[i,j])*(3*sigmaj)*(3*sigmai) + Fmatlen[i,i]*(3*sigmai)**2))
            #print('z3, z1',z3, z1)
            cs = axes[i,j].contourf(X,Y,Z,[z3, z1, 1], colors=['yellowgreen', 'olive', 'honeydew'])
            cs.cmap.set_over('blue')
            axes[i,j].set_ylim([yst, yed])
            axes[i,j].set_xlim([xst, xed])
            print('i, j', i,j ,p1, p2)
            print('xst, yst', xst, yst)

            if j == 0:
                axes[i,j].set_ylabel(p1)
            if i == pnum-1:
                axes[i,j].set_xlabel(p2)
plt.subplots_adjust(wspace=0, hspace=0)
#plt.tight_layout()
plt.savefig('compare_CDMp_cov_noise6_phi_dl%s_%s_len.png'%(dl, camborself))



'''





