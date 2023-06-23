import numpy as np, scipy as sc, sys, argparse, os
import tools
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy import interpolate 
from scipy.interpolate import interp1d
from os.path import exists
import json
import pandas as pd
from datetime import datetime
from matplotlib.animation import FuncAnimation
import re
from scipy.stats.distributions import chi2

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
param_dict = tools.get_ini_param_dict(paramfile)


para_value = {}
prior_dic = {'tau':0.001}
#prior_dic1 = {'gamma_phi_sys':1}
#prior_dic = {'tau':0.007}
prior_dic = {}


which_spectra = 'delensed_scalar'
sysornot = 'sys'
itername = 'iter1st'
itername = 'iter0'
#rms_map_T_list = np.arange(1,11,1)
rms_map_T_list = np.array((0.1,0.2,0.5, 1.0, 2.0 ,5.0, 10.0))#np.arange(1,11,1)


dl = 5
binsize = 5
systype = 'gphionly'

clname = ['TT','EE','TE','BB','PP']
els = np.arange(2,5001)
newl = np.arange(els[0], els[-1]+1, 5)

param_list = ['As', 'mnu', 'neff', 'ns', 'ombh2', 'omch2', 'r','tau', 'thetastar']
param_list_sys = ['As', 'gamma_phi_sys', 'mnu', 'neff', 'ns', 'ombh2', 'omch2', 'r','tau', 'thetastar']

for item in param_list_sys:
    para_value[item] = param_dict[item]

param_names = param_list
name = ['TTTT','EEEE','TETE','BBBB','PPPP']



fsky = 0.57
Lsdl = 5
i = 0

len_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list), len(param_list)))
unlen_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list), len(param_list)))
len_sigma_array_bin5_inv_noncov = np.zeros((len(rms_map_T_list), len(param_list)))
unlen_sigma_array_bin5_inv_noncov = np.zeros((len(rms_map_T_list), len(param_list)))
len_sigma_array_bin5_inv_gaussian = np.zeros((len(rms_map_T_list), len(param_list)))
unlen_sigma_array_bin5_inv_gaussian = np.zeros((len(rms_map_T_list), len(param_list)))
len_sigma_array_bin5_inv_nomnu = np.zeros((len(rms_map_T_list), len(param_list)-1))
unlen_sigma_array_bin5_inv_nomnu = np.zeros((len(rms_map_T_list), len(param_list)-1))
len_sigma_array_bin5_inv_gaussian_nomnu = np.zeros((len(rms_map_T_list), len(param_list)-1))
unlen_sigma_array_bin5_inv_gaussian_nomnu = np.zeros((len(rms_map_T_list), len(param_list)-1))


delen_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array_bin5_inv_noncov = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array_bin5_inv_gaussian = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array_bin5_inv_nomnu = np.zeros((len(rms_map_T_list), len(param_list)-1))
delen_sigma_array_bin5_inv_gaussian_nomnu = np.zeros((len(rms_map_T_list), len(param_list)-1))

delensys_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list), 1+len(param_list)))
delensys_sigma_array_bin5_inv_noncov = np.zeros((len(rms_map_T_list), 1+len(param_list)))
delensys_sigma_array_bin5_inv_gaussian = np.zeros((len(rms_map_T_list), 1+len(param_list)))
delensys_sigma_array_bin5_inv_nomnu = np.zeros((len(rms_map_T_list), len(param_list)))
delensys_sigma_array_bin5_inv_gaussian_nomnu = np.zeros((len(rms_map_T_list), len(param_list)))



dl = 5
camborself = "self"
derivname = "selfdriv"#"selfdriv"
camborself = "self"
binsize = 5
Lsdl = dl

addlensing = False
#addlensing = True
addBB = False
addBB = True

if addlensing == False and addBB == False:
    tail = 'TTEETE'
if addlensing == True and addBB == True:
    tail = 'addlensing'
elif addlensing == False and addBB == True:
    tail = 'noPhi'
elif addBB == False and addlensing == True:
    tail = 'noBB'


for i, item in enumerate(rms_map_T_list):

    Fmatunlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_%s.npy"%('zeroN0','unlensed_total', 'compall', rms_map_T_list[i], tail))
    Fmatunlen = tools.add_prior(Fmatunlen, param_names, prior_dic)
    Fmatunlennomnu = np.delete(Fmatunlen, (1,), axis = 0)
    Fmatunlennomnu = np.delete(Fmatunlennomnu, (1,), axis = 1)

    unlen_sigma_array_bin5_inv[i] = (np.diag(np.matrix(Fmatunlen).I))**0.5
    unlen_sigma_array_bin5_inv_noncov[i] = (1/np.diag(Fmatunlen))**0.5
    unlen_sigma_array_bin5_inv_nomnu[i] = (np.diag(np.matrix(Fmatunlennomnu).I))**0.5

    Fmatunlengauss = np.load("ForNote/F_mat_%s_%s%s_n%s_%s.npy"%('zeroN0','unlensed_total', 'compall', rms_map_T_list[i], tail))
    Fmatunlengauss = tools.add_prior(Fmatunlengauss, param_names, prior_dic)
    Fmatunlengaussnomnu = np.delete(Fmatunlengauss, (1,), axis = 0)
    Fmatunlengaussnomnu = np.delete(Fmatunlengaussnomnu, (1,), axis = 1)
    unlen_sigma_array_bin5_inv_gaussian[i] = (np.diag(np.matrix(Fmatunlengauss).I))**0.5
    unlen_sigma_array_bin5_inv_gaussian_nomnu[i] = (np.diag(np.matrix(Fmatunlengaussnomnu).I))**0.5

for i, item in enumerate(rms_map_T_list):

        Fmatlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_%s.npy"%('zeroN0', 'total','compall', rms_map_T_list[i], tail))
        Fmatlen = tools.add_prior(Fmatlen, param_names, prior_dic)
        len_sigma_array_bin5_inv[i] = (np.diag(np.matrix(Fmatlen).I))**0.5
        Fmatlennomnu = np.delete(Fmatlen, (1,), axis = 0)
        Fmatlennomnu = np.delete(Fmatlennomnu, (1,), axis = 1)
        len_sigma_array_bin5_inv_nomnu[i] = (np.diag(np.matrix(Fmatlennomnu).I))**0.5
        Fmatlengauss = np.load("ForNote/F_mat_%s_%s%s_n%s_%s.npy"%('zeroN0', 'total','compall', rms_map_T_list[i], tail))
        Fmatlengauss = tools.add_prior(Fmatlengauss, param_names, prior_dic)
        Fmatlengaussnomnu = np.delete(Fmatlengauss, (1,), axis = 0)
        Fmatlengaussnomnu = np.delete(Fmatlengaussnomnu, (1,), axis = 1)
        len_sigma_array_bin5_inv_gaussian[i] = (np.diag(np.matrix(Fmatlengauss).I))**0.5
        len_sigma_array_bin5_inv_gaussian_nomnu[i] = (np.diag(np.matrix(Fmatlengaussnomnu).I))**0.5
        len_sigma_array_bin5_inv_noncov[i] = (1/np.diag(Fmatlen))**0.5

        Fmatdelensys = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_%s.npy"%(itername,'delensed_scalar','compall' , rms_map_T_list[i], tail))
        Fmatdelen = np.delete(Fmatdelensys, (1,), axis = 0)
        Fmatdelen = np.delete(Fmatdelen, (1,), axis = 1)
        Fmatdelen = tools.add_prior(Fmatdelen, param_names, prior_dic)
        Fmatdelennomnu = np.delete(Fmatdelen, (1,), axis = 0)
        Fmatdelennomnu = np.delete(Fmatdelennomnu, (1,), axis = 1)
        delen_sigma_array_bin5_inv_nomnu[i] = (np.diag(np.matrix(Fmatdelennomnu).I))**0.5
        print(Fmatdelen.shape)
        delen_sigma_array_bin5_inv[i] = (np.diag(np.matrix(Fmatdelen).I))**0.5
        delen_sigma_array_bin5_inv_noncov[i] = (1/np.diag(Fmatdelen))**0.5
        Fmatdelensysgauss = np.load("ForNote/F_mat_%s_%s%s_n%s_%s.npy"%(itername,'delensed_scalar','compall' , rms_map_T_list[i], tail))
        Fmatdelengauss = np.delete(Fmatdelensysgauss, (1,), axis = 1)
        Fmatdelengauss = np.delete(Fmatdelengauss, (1,), axis = 0)
        Fmatdelengauss = tools.add_prior(Fmatdelengauss, param_list, prior_dic)
        delen_sigma_array_bin5_inv_gaussian[i] = (np.diag(np.matrix(Fmatdelengauss).I))**0.5
        Fmatdelengaussnomnu = np.delete(Fmatdelengauss, (1,), axis = 0)
        Fmatdelengaussnomnu = np.delete(Fmatdelengaussnomnu, (1,), axis = 1)
        delen_sigma_array_bin5_inv_gaussian_nomnu[i] = (np.diag(np.matrix(Fmatdelengaussnomnu).I))**0.5
        Fmatdelensys = tools.add_prior(Fmatdelensys, param_list_sys, prior_dic)
        Fmatdelensysgauss = tools.add_prior(Fmatdelensysgauss, param_list_sys, prior_dic)
        delensys_sigma_array_bin5_inv_gaussian[i] = (np.diag(np.matrix(Fmatdelensysgauss).I))**0.5
        delensys_sigma_array_bin5_inv[i] = (np.diag(np.matrix(Fmatdelensys).I))**0.5
        Fmatdelensysnomnu = np.delete(Fmatdelensys, (2,), axis = 0)
        Fmatdelensysnomnu = np.delete(Fmatdelensysnomnu, (2,), axis = 1)
        delensys_sigma_array_bin5_inv_nomnu[i] = (np.diag(np.matrix(Fmatdelensysnomnu).I))**0.5
        Fmatdelensysgaussnomnu = np.delete(Fmatdelensysgauss, (2,), axis = 0)
        Fmatdelensysgaussnomnu = np.delete(Fmatdelensysgaussnomnu, (2,), axis = 1)
        delensys_sigma_array_bin5_inv_gaussian_nomnu[i] = (np.diag(np.matrix(Fmatdelensysgaussnomnu).I))**0.5

delensys_sigma_array_bin5_inv_short = np.column_stack((delensys_sigma_array_bin5_inv[:,0], delensys_sigma_array_bin5_inv[:,2:]))
delensys_sigma_array_bin5_inv_gaussian_short = np.column_stack((delensys_sigma_array_bin5_inv_gaussian[:,0], delensys_sigma_array_bin5_inv_gaussian[:,2:]))


plt.figure(figsize = (12,12))
for i, item in enumerate(param_list):
    axi = plt.subplot(3,3,i+1)
    axi.loglog(rms_map_T_list, len_sigma_array_bin5_inv[:,i],'blue', label = 'lensed')
    axi.loglog(rms_map_T_list, len_sigma_array_bin5_inv_gaussian[:,i],'blue', linestyle = '-.',label = 'lensed gaussian')
    axi.loglog(rms_map_T_list, unlen_sigma_array_bin5_inv[:,i],'red', label = 'unlensed')
    axi.loglog(rms_map_T_list, unlen_sigma_array_bin5_inv_gaussian[:,i],'red', linestyle = '-.',label = 'unlensed gaussian')
    axi.loglog(rms_map_T_list, delen_sigma_array_bin5_inv[:,i],'orange', label = 'delensed')
    axi.loglog(rms_map_T_list, delen_sigma_array_bin5_inv_gaussian[:,i],'orange', label = 'delensed gaussian', linestyle = '-.',) #'nomarginalization'
    axi.loglog(rms_map_T_list, delensys_sigma_array_bin5_inv_short[:,i],'green', label = 'delensedsys')
    axi.loglog(rms_map_T_list, delensys_sigma_array_bin5_inv_gaussian_short[:,i],'green', label = 'delensedsys gaussian', linestyle = '-.') 
    axi.legend()
    #axi.set_xlim(0,1)

    axi.set_title(param_list[i])

plt.tight_layout()
plt.savefig('ForNote/compare_alls_%s_%s_dl%s_%s_rem6_noprior.png'%(itername, tail, dl, camborself))


delensys_sigma_array_bin5_inv_nomnu_short = np.column_stack((delensys_sigma_array_bin5_inv_nomnu[:,0], delensys_sigma_array_bin5_inv_nomnu[:,2:]))
delensys_sigma_array_bin5_inv_gaussian_nomnu_short = np.column_stack((delensys_sigma_array_bin5_inv_gaussian_nomnu[:,0], delensys_sigma_array_bin5_inv_gaussian_nomnu[:,2:]))



param_list_nomun = ['As', 'neff', 'ns', 'ombh2', 'omch2', 'r','tau', 'thetastar']

plt.figure(figsize = (12,12))
for i, item in enumerate(param_list_nomun):
    axi = plt.subplot(3,3,i+1)
    axi.loglog(rms_map_T_list, len_sigma_array_bin5_inv_nomnu[:,i],'blue', label = 'lensed')
    axi.loglog(rms_map_T_list, len_sigma_array_bin5_inv_gaussian_nomnu[:,i],'blue', linestyle = '-.',label = 'lensed gaussian')
    axi.loglog(rms_map_T_list, unlen_sigma_array_bin5_inv_nomnu[:,i],'red', label = 'unlensed')
    axi.loglog(rms_map_T_list, unlen_sigma_array_bin5_inv_gaussian_nomnu[:,i],'red', linestyle = '-.',label = 'unlensed gaussian')
    axi.loglog(rms_map_T_list, delen_sigma_array_bin5_inv_nomnu[:,i],'orange', label = 'delensed')
    axi.loglog(rms_map_T_list, delen_sigma_array_bin5_inv_gaussian_nomnu[:,i],'orange', label = 'delensed gaussian', linestyle = '-.',) #'nomarginalization'
    axi.loglog(rms_map_T_list, delensys_sigma_array_bin5_inv_nomnu_short[:,i],'green', label = 'delensedsys')
    axi.loglog(rms_map_T_list, delensys_sigma_array_bin5_inv_gaussian_nomnu_short[:,i],'green', label = 'delensedsys gaussian', linestyle = '-.') 
    axi.legend()
    #axi.set_xlim(0,1)

    axi.set_title(param_list_nomun[i])

axi = plt.subplot(3,3, i+2)
axi.semilogx(rms_map_T_list, delensys_sigma_array_bin5_inv_nomnu[:,1],'green', label='$\gamma_{\phi}$')
axi.semilogx(rms_map_T_list, delensys_sigma_array_bin5_inv_gaussian_nomnu[:,1],'green', label='$\gamma_{\phi}$', linestyle = '-.')
axi.legend()
axi.set_title("gamma")

plt.tight_layout()
plt.savefig('ForNote/compare_nomnu_%s_%s_dl%s_%s_rem6_noprior.png'%(itername, tail, dl, camborself))

systype = 'compall'
ni = -1
Fmatlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_%s.npy"%('zeroN0',"total", systype, rms_map_T_list[ni], tail))
Fmatunlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_%s.npy"%('zeroN0',"unlensed_total", systype, rms_map_T_list[ni], tail))
Fmatdelensys = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_%s.npy"%('iter0',"delensed_scalar", systype, rms_map_T_list[ni], tail))

Fmatdelen = np.delete(Fmatdelensys, (1,), axis = 0)
Fmatdelen = np.delete(Fmatdelen, (1,), axis = 1)

Fmatlen = tools.add_prior(Fmatlen, param_names, prior_dic)
Fmatunlen = tools.add_prior(Fmatunlen, param_names, prior_dic)
Fmatdelen = tools.add_prior(Fmatdelen, param_names, prior_dic)

cov_matdelensys0 = np.matrix(Fmatdelensys).I
cov_matdelen0 = np.matrix(Fmatdelen).I
cov_matlen0 = np.matrix(Fmatlen).I
cov_matunlen0 = np.matrix(Fmatunlen).I


pnum = len(param_list_sys) - 1

z1 = np.exp(-chi2.ppf(0.683, df=2)/2)
z3 = np.exp(-chi2.ppf(0.997, df=2)/2)

plt.clf()
fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum, nrows=pnum)
for i, p1 in enumerate(param_list):
    for j, p2 in enumerate(param_list):
        if i<j:
            axes[i,j].axis('off')
            #for j, p2 in enumerate(param_list[:i+1]):        
        elif i==j:
            sigmai = cov_matdelen0[i,i]**0.5
            sigmailen = cov_matlen0[i,i]**0.5
            sigmaiunlen = cov_matunlen0[i,i]**0.5
            valuei = para_value[p1]
            xst = valuei - sigmai*5
            xed = valuei + sigmai*5
            x = np.linspace(xst, xed, 200)            
            coe = 1.0 / (2 * np.pi)**0.5 / sigmai
            Z = np.exp (-0.5 * ((x-valuei)/sigmai)**2)
            Zlen = np.exp (-0.5 * ((x-valuei)/sigmailen)**2)
            Zunlen = np.exp (-0.5 * ((x-valuei)/sigmaiunlen)**2)
            axes[i,i].plot(x, Z, 'green')
            axes[i,i].plot(x, Zlen, 'blue')
            axes[i,i].plot(x, Zunlen, 'orange')
            print('i = ',i)
            #axes[i,j].xlim
        else:
            sigmai = cov_matdelen0[i,i]**0.5
            sigmaj = cov_matdelen0[j,j]**0.5
            valuei = para_value[p1]
            valuej = para_value[p2]
            sigmailen = cov_matlen0[i,i]**0.5
            sigmajlen = cov_matlen0[j,j]**0.5
            sigmaiunlen = cov_matunlen0[i,i]**0.5
            sigmajunlen = cov_matunlen0[j,j]**0.5
            xst = valuej - sigmaj*5
            xed = valuej + sigmaj*5
            yst = valuei - sigmai*5
            yed = valuei + sigmai*5
            x = np.linspace(xst, xed, 80)
            y = np.linspace(yst, yed, 80)
            X,Y = np.meshgrid(x,y)

            Fmatdelen0 = np.matrix([[cov_matdelen0[j,j], cov_matdelen0[j,i]],[cov_matdelen0[i,j],cov_matdelen0[i,i]]]).I
            Fmatlen0 = np.matrix([[cov_matlen0[j,j], cov_matlen0[j,i]],[cov_matlen0[i,j],cov_matlen0[i,i]]]).I
            Fmatunlen0 = np.matrix([[cov_matunlen0[j,j], cov_matunlen0[j,i]],[cov_matunlen0[i,j],cov_matunlen0[i,i]]]).I

            Zlen = np.exp(-0.5 * (Fmatlen0[0,0]*(X-valuej)**2 + (Fmatlen0[0,1] + Fmatlen0[1,0])*(X-valuej)*(Y-valuei) + Fmatlen0[1,1]*(Y-valuei)**2))
            Zunlen = np.exp(-0.5 * (Fmatunlen0[0,0]*(X-valuej)**2 + (Fmatunlen0[0,1] + Fmatunlen0[1,0])*(X-valuej)*(Y-valuei) + Fmatunlen0[1,1]*(Y-valuei)**2))
            Zdelen = np.exp(-0.5 * (Fmatdelen0[0,0]*(X-valuej)**2 + (Fmatdelen0[0,1] + Fmatdelen0[1,0])*(X-valuej)*(Y-valuei) + Fmatdelen0[1,1]*(Y-valuei)**2))
            #print('z3, z1',z3, z1)
            cs = axes[i,j].contourf(X,Y,Zdelen,[z3, z1, 1], colors=['yellowgreen', 'olive', 'honeydew'])
            cslen = axes[i,j].contour(X,Y,Zlen,[z3, 1], colors=['blue', 'yellow'])
            csunlen = axes[i,j].contour(X,Y,Zunlen,[z3, 1], colors=['orange', 'yellow'])
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
plt.savefig('compare_%s_%s_dl%s_compall_noprior.png'%(rms_map_T_list[ni], tail, dl))

