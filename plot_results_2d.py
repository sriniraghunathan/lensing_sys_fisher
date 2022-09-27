import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np, scipy as sc, sys, argparse, os
from scipy.stats.distributions import chi2

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
binsize = 5
noisei = 0
alpha1 = 1.52
alpha3 = 3.44

with open("results/F_mat_CDMp_test_%s_bin%s_dl%s_%s_n%s.json"%("delensed_scalar", binsize, Lsdl, camborself, rms_map_T_list[noisei])) as infile:
    F_matdelen = json.load(infile)
with open("results/F_mat_CDMp_test_%s_bin%s_dl%s_%s_n%s.json"%("total", binsize, Lsdl, camborself, rms_map_T_list[noisei])) as infile:
    F_matlen = json.load(infile)
with open("results/F_mat_CDMp_test_%s_bin%s_dl%s_%s_n%s.json"%("unlensed_total", 5, 5, "self", rms_map_T_list[noisei])) as infile:
    F_matunlen = json.load(infile)
with open("results/F_mat_CDMp_prior3_lensys_%s_bin%s_dl%s_%s_n%s.json"%("delensed_scalar", binsize, Lsdl, camborself, rms_map_T_list[noisei])) as infile:
    F_matdelensys = json.load(infile)
with open("results/F_mat_CDMp_prior_lensys_%s_bin%s_dl%s_%s_n%s.json"%("delensed_scalar", binsize, Lsdl, camborself, rms_map_T_list[noisei])) as infile:
    F_matdelensysnp = json.load(infile)

#results/F_mat_CDMp_test_%s_bin%s_dl%s_%s_n%s.json
logline = '\tget/read the parameter file'; tools.write_log(logline)
param_dict = tools.get_ini_param_dict(paramfile)

param_list = F_matdelen['parms']
paramsys_list = F_matdelensys['parms']

para_value = {}

for item in paramsys_list:
    para_value[item] = param_dict[item]


Fmatdelen = np.asarray(F_matdelen['Fmat'])
cov_matdelen = np.asarray(F_matdelen['cov_mat'])
Fmatlen = np.asarray(F_matlen['Fmat'])
cov_matlen = np.asarray(F_matlen['cov_mat'])
Fmatunlen = np.asarray(F_matunlen['Fmat'])
cov_matunlen = np.asarray(F_matunlen['cov_mat'])
Fmatdelensys = np.asarray(F_matdelensys['Fmat'])
cov_matdelensys = np.asarray(F_matdelensys['cov_mat'])
Fmatdelensysnp = np.asarray(F_matdelensysnp['Fmat'])
cov_matdelensysnp = np.asarray(F_matdelensysnp['cov_mat'])

pnum = len(param_list)
pnumsys = len(paramsys_list)

print(cov_matlen)

#plt.figure(figsize = (12,12))
#fig, axes = plt.subplots(figsize=(10, 10), sharex = 'col',  sharey = 'row', ncols=pnum, nrows=pnum)
fig, axes = plt.subplots(figsize=(16, 16), ncols=pnumsys, nrows=pnumsys)
#fig = plt.figure()
#gs = fig.add_gridspec(pnum, pnum, hspace=0, wspace=0)
#axes = plt.subplots()
#gs = fig.add_gridspec(hspace=0, wspace=0)

for i, p1 in enumerate(paramsys_list):
    for j, p2 in enumerate(paramsys_list):
        if i<j:
            axes[i,j].axis('off')
            #for j, p2 in enumerate(param_list[:i+1]):        
        elif i==j:
            sigmaidelensys = cov_matdelensys[i,i]**0.5
            sigmaidelensysnp = cov_matdelensysnp[i,i]**0.5
            valuei = para_value[p1]
            xst = valuei - sigmaidelensys*4
            xed = valuei + sigmaidelensys*4
            x = np.linspace(xst, xed, 40)            
            coe = 1.0 / (2 * np.pi)**0.5 / sigmaidelensys
            Z = np.exp (-0.5 * ((x-valuei)/sigmaidelensys)**2)
            Znp = np.exp (-0.5 * ((x-valuei)/sigmaidelensysnp)**2)
            axes[i,i].plot(x, Z, 'orange')
            axes[i,i].plot(x, Znp, 'navy')
            print('i = ',i)
            if i == pnumsys-1:
                axes[i,j].set_xlabel(p2)
            if i == 0:
                axes[i,j].set_ylabel(p1)
            #axes[i,j].xlim
        else:
            sigmaidelensys = cov_matdelensys[i,i]**0.5
            sigmajdelensys = cov_matdelensys[j,j]**0.5
            sigmaidelensysnp = cov_matdelensysnp[i,i]**0.5
            sigmajdelensysnp = cov_matdelensysnp[j,j]**0.5
            #covmat = np.matrix([[Fmatdelensys[i,i], Fmatdelensys[i,j]],[Fmatdelensys[j,i], Fmatdelensys[j,j]]]).I
            valuei = para_value[p1]
            valuej = para_value[p2]
            xst = valuej - sigmajdelensys*5
            xed = valuej + sigmajdelensys*5
            yst = valuei - sigmaidelensys*5
            yed = valuei + sigmaidelensys*5
            x = np.linspace(xst, xed, 80)
            y = np.linspace(yst, yed, 80)
            X,Y = np.meshgrid(x,y)
            #coe = 1.0 / ((2 * np.pi)**2 * np.linalg.det(cov_matlen))**0.5
            Fmatdelensys = np.matrix([[cov_matdelensys[j,j], cov_matdelensys[j,i]],[cov_matdelensys[i,j],cov_matdelensys[i,i]]]).I
            Z = np.exp(-0.5 * (Fmatdelensys[0,0]*(X-valuej)**2 + (Fmatdelensys[0,1] + Fmatdelensys[1,0])*(X-valuej)*(Y-valuei) + Fmatdelensys[1,1]*(Y-valuei)**2))
            #z1s = np.exp(-0.5 * (Fmatdelensys[0,0]*(sigmajdelensys*alpha1)**2 + (Fmatdelensys[0,1] + Fmatdelensys[1,0])*(sigmajdelensys*alpha1)*(Y-valuei) + Fmatdelensys[1,1]*(Y-valuei)**2))
            #z1 = np.max(z1s)
            z1 = np.exp(-chi2.ppf(0.683, df=2)/2)
            z3 = np.exp(-chi2.ppf(0.997, df=2)/2)
            Fmatdelensysnp = np.matrix([[cov_matdelensysnp[j,j], cov_matdelensysnp[j,i]],[cov_matdelensysnp[i,j],cov_matdelensysnp[i,i]]]).I
            Znp = np.exp(-0.5 * (Fmatdelensysnp[0,0]*(X-valuej)**2 + (Fmatdelensysnp[0,1] + Fmatdelensysnp[1,0])*(X-valuej)*(Y-valuei) + Fmatdelensysnp[1,1]*(Y-valuei)**2))
            z1np = z1
            z3np = z3
            #Zdelensys = np.exp(-0.5 * (Fmatdelensys[j,j]*(X-valuej)**2 + (Fmatdelensys[i,j] + Fmatdelensys[i,j])*(X-valuej)*(Y-valuei) + Fmatdelensys[i,i]*(Y-valuei)**2))

            #cs = axes[i,j].contourf(X,Y,Z,[z3, z1, 1], colors=['yellowgreen', 'olive', 'honeydew'])
            csdelensys = axes[i,j].contourf(X,Y,Z,[z3, z1, 1], colors=['orange', 'yellow'])
            csdelensys = axes[i,j].contour(X,Y,Znp,[z3np, z1np, 1], colors=['navy', 'royalblue'])

            axes[i,j].set_ylim([yst, yed])
            axes[i,j].set_xlim([xst, xed])
            print('i, j', i,j ,p1, p2)
            print('xst, yst', xst, yst)

            if j == 0:
                axes[i,j].set_ylabel(p1)
            if i == pnumsys-1:
                axes[i,j].set_xlabel(p2)
plt.subplots_adjust(wspace=0, hspace=0)
#plt.tight_layout()
plt.savefig('compare_CDMp_lensys2_cov_noise%s_phi_dl%s_%s_delen.png'%(rms_map_T_list[noisei],dl, camborself))



plt.clf()
fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum, nrows=pnum)
for i, p1 in enumerate(param_list):
    for j, p2 in enumerate(param_list):
        if i<j:
            axes[i,j].axis('off')
            #for j, p2 in enumerate(param_list[:i+1]):        
        elif i==j:
            sigmai = cov_matdelen[i,i]**0.5
            sigmailen = cov_matlen[i,i]**0.5
            sigmaiunlen = cov_matunlen[i,i]**0.5
            valuei = para_value[p1]
            xst = valuei - sigmai*5
            xed = valuei + sigmai*5
            x = np.linspace(xst, xed, 40)            
            coe = 1.0 / (2 * np.pi)**0.5 / sigmai
            Z = np.exp (-0.5 * ((x-valuei)/sigmai)**2)
            Zlen = np.exp (-0.5 * ((x-valuei)/sigmailen)**2)
            Zunlen = np.exp (-0.5 * ((x-valuei)/sigmaiunlen)**2)
            axes[i,i].plot(x, Z, 'green')
            axes[i,i].plot(x, Zlen, 'blue')
            axes[i,i].plot(x, Zunlen, 'orange')
            print('i = ',i)
            if i == 0:
                axes[i,j].set_ylabel(p1)

            #axes[i,j].xlim
        else:
            sigmai = cov_matdelen[i,i]**0.5
            sigmaj = cov_matdelen[j,j]**0.5
            valuei = para_value[p1]
            valuej = para_value[p2]
            sigmailen = cov_matlen[i,i]**0.5
            sigmajlen = cov_matlen[j,j]**0.5
            sigmaiunlen = cov_matunlen[i,i]**0.5
            sigmajunlen = cov_matunlen[j,j]**0.5
            xst = valuej - sigmaj*5
            xed = valuej + sigmaj*5
            yst = valuei - sigmai*5
            yed = valuei + sigmai*5
            x = np.linspace(xst, xed, 80)
            y = np.linspace(yst, yed, 80)
            X,Y = np.meshgrid(x,y)
            coe = 1.0 / ((2 * np.pi)**2 * np.linalg.det(cov_matlen))**0.5
            Fmatdelen = np.matrix([[cov_matdelen[j,j], cov_matdelen[j,i]],[cov_matdelen[i,j],cov_matdelen[i,i]]]).I
            Fmatlen = np.matrix([[cov_matlen[j,j], cov_matlen[j,i]],[cov_matlen[i,j],cov_matlen[i,i]]]).I
            Fmatunlen = np.matrix([[cov_matunlen[j,j], cov_matunlen[j,i]],[cov_matunlen[i,j],cov_matunlen[i,i]]]).I


            Z = np.exp(-0.5 * (Fmatdelen[0,0]*(X-valuej)**2 + (Fmatdelen[0,1] + Fmatdelen[1,0])*(X-valuej)*(Y-valuei) + Fmatdelen[1,1]*(Y-valuei)**2))
            z1s = np.exp(-0.5 * (Fmatdelen[0,0]*(sigmaj*alpha1)**2 + (Fmatdelen[0,1] + Fmatdelen[1,0])*(sigmaj*alpha1)*(Y-valuei) + Fmatdelen[1,1]*(Y-valuei)**2))
            z1 = np.max(z1s)
            z3s = np.exp(-0.5 * (Fmatdelen[0,0]*(sigmaj*alpha3)**2 + (Fmatdelen[0,1] + Fmatdelen[1,0])*(sigmaj*alpha3)*(Y-valuei) + Fmatdelen[1,1]*(Y-valuei)**2))
            z3 = np.max(z3s)
            z1 = np.exp(-chi2.ppf(0.683, df=2)/2)
            z3 = np.exp(-chi2.ppf(0.997, df=2)/2)

            #z1 = np.exp(-0.5 * (Fmatdelen[0,0]*(sigmaj)**2 + (Fmatdelen[0,1] + Fmatdelen[1,0])*(sigmaj)*(sigmai) + Fmatdelen[1,1]*(sigmai)**2))

            Zlen = np.exp(-0.5 * (Fmatlen[0,0]*(X-valuej)**2 + (Fmatlen[0,1] + Fmatlen[1,0])*(X-valuej)*(Y-valuei) + Fmatlen[1,1]*(Y-valuei)**2))
            #z1lens = np.exp(-0.5 * (Fmatlen[0,0]*(sigmajlen*alpha1)**2 + (Fmatlen[0,1] + Fmatlen[1,0])*(sigmajlen*alpha1)*(Y-valuei) + Fmatlen[1,1]*(Y-valuei)**2))
            #z1len = np.max(z1lens)
            #z1len = np.exp(-0.5 * (Fmatlen[0,0]*(sigmajlen)**2 + (Fmatlen[0,1] + Fmatlen[1,0])*(sigmajlen)*(sigmailen) + Fmatlen[1,1]*(sigmailen)**2))
            Zunlen = np.exp(-0.5 * (Fmatunlen[0,0]*(X-valuej)**2 + (Fmatunlen[0,1] + Fmatunlen[1,0])*(X-valuej)*(Y-valuei) + Fmatunlen[1,1]*(Y-valuei)**2))

            #print('z3, z1',z3, z1)
            cs = axes[i,j].contourf(X,Y,Z,[z3, z1, 1], colors=['yellowgreen', 'olive', 'honeydew'])
            cslen = axes[i,j].contour(X,Y,Zlen,[z1, 1], colors=['blue', 'yellow'])
            csunlen = axes[i,j].contour(X,Y,Zunlen,[z1, 1], colors=['orange', 'yellow'])
            #cs.cmap.set_over('blue')
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
plt.savefig('compare_CDMp_test_chi2_noise%s_phi_dl%s_%s_delen.png'%(rms_map_T_list[noisei],dl, camborself))




