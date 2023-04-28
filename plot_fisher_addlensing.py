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
#prior_dic = {}




which_spectra = 'delensed_scalar'
sysornot = 'sys'
itername = 'iter1st'
rms_map_T_list = np.array((0.1, 0.5))#np.arange(1,11,1)
rms_map_T_list = np.array((0.1,0.2,0.5, 1.0, 2.0 ,5.0, 10.0))#np.arange(1,11,1)
rms_map_T_list_long = np.array((0.1,0.2,0.5, 1.0, 2.0 ,5.0, 10.0))#np.arange(1,11,1)

dl = 5
binsize = 5
systype = 'guessw'
systype = 'updateall'
systype = 'changefield'

clname = ['TT','EE','TE','BB','PP']
els = np.arange(2,5001)
newl = np.arange(els[0], els[-1]+1, 5)

param_list_sys = ['As', 'gamma_N0_sys', 'gamma_phi_sys', 'neff','ns', 'ombh2', 'omch2', 'r', 'tau', 'thetastar']
param_list = ['As', 'neff', 'ns', 'ombh2', 'omch2', 'r', 'tau', 'thetastar']
delensysparam_list = param_list_sys

for item in param_list_sys:
    para_value[item] = param_dict[item]

param_names = param_list
name = ['TTTT','EEEE','TETE','BBBB','PPPP']
delensys_sigma_array = np.zeros((len(rms_map_T_list), len(param_list)+2))
delen_sigma_array = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array_noncov = np.zeros((len(rms_map_T_list), len(param_list)))
prior_dic = {'tau':0.001}




Lsdl = 5
i = 0

len_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list_long), len(param_list)))
unlen_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list), len(param_list)))
len_sigma_array_bin5_inv_noncov = np.zeros((len(rms_map_T_list_long), len(param_list)))
unlen_sigma_array_bin5_inv_noncov = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array_bin5_inv2 = np.zeros((len(rms_map_T_list), len(param_list)))
len_sigma_array = np.zeros((len(rms_map_T_list_long), len(param_list)))

#delensys_sigma_array_bin5_inv = np.zeros((len(rms_map_T_list), len(param_list)+2))
delensys_sigma_array_bin5_inv1 = np.zeros((len(rms_map_T_list_long), len(param_list)+2))
delensys_sigma_array_bin5_inv2 = np.zeros((len(rms_map_T_list_long), len(param_list)+2))
delensys_sigma_array_bin5_long = np.zeros((len(rms_map_T_list_long), len(param_list)+2))
delen_sigma_array_bin5_long = np.zeros((len(rms_map_T_list_long), len(param_list)))
delensys_sigma_array_bin5_long_noncov = np.zeros((len(rms_map_T_list_long), len(param_list)+2))
delen_sigma_array_bin5_long_noncov = np.zeros((len(rms_map_T_list_long), len(param_list)))


dl = 5
camborself = "self"
derivname = "selfdriv"#"selfdriv"
camborself = "self"
prior_dic = {'tau':0.001}
prior_dic1 = {'gamma_phi_sys':1}
prior_dic2 = {'gamma_N0_sys':1}
prior_dic3 = {'gamma_phi_sys':1, 'gamma_N0_sys':1}
#prior_dic3 = {'gamma_phi_sys':0.001}
#prior_dic = {'tau':0.007}
#prior_dic = {'tau':0.007, 'A_phi_sys':5e-19, 'alpha_phi_sys':0.2}
binsize = 5
Lsdl = dl

for i, item in enumerate(rms_map_T_list):
    with open("results/F_mat_CDMp_%s%s_bin%s_dl%s_%s_n%s_2gammas1.00_rem6_%s_addlensing2.json"%('zeroN0', "unlensed_total", binsize, dl, 'self', rms_map_T_list[i], systype)) as infile:
        F_matunlen = json.load(infile)

    Fmatunlen = np.asarray(F_matunlen['Fmat'])

    Fmatunlen = tools.add_prior(Fmatunlen, param_names, prior_dic)
    #Fmatlen = tools.add_prior(Fmatlen, param_names, prior_dic)

    unlen_sigma_array_bin5_inv[i] = (np.diag(np.matrix(Fmatunlen).I))**0.5
    #len_sigma_array_bin5_inv[i] = (np.diag(np.matrix(Fmatlen).I))**0.5
    unlen_sigma_array_bin5_inv_noncov[i] = (1/np.diag(Fmatunlen))**0.5
    #len_sigma_array_bin5_inv_noncov[i] = (1/np.diag(Fmatlen))**0.5

for i, item in enumerate(rms_map_T_list_long):
    with open("results/F_mat_CDMp_%s%s_bin%s_dl%s_%s_n%s_2gammas1.00_rem6_%s_addlensing2.json"%('iter1st',"delensed_scalar", binsize, Lsdl, camborself, item, systype)) as infile:
        F_matdelensyslong = json.load(infile)
    with open("results/F_mat_CDMp_%s%s_bin%s_dl%s_%s_n%s_2gammas1.00_rem6_%s_addlensing2.json"%('zeroN0',"total", binsize, Lsdl, camborself, item, systype)) as infile:
        F_matlen = json.load(infile)

        Fmatlen = np.asarray(F_matlen['Fmat'])
        Fmatlen = tools.add_prior(Fmatlen, param_names, prior_dic)
        len_sigma_array_bin5_inv[i] = (np.diag(np.matrix(Fmatlen).I))**0.5
        len_sigma_array_bin5_inv_noncov[i] = (1/np.diag(Fmatlen))**0.5

        Fmatdelensyslong = np.asarray(F_matdelensyslong['Fmat'])
        Fmatdelensyslong = tools.add_prior(Fmatdelensyslong, delensysparam_list, prior_dic)
        delensys_sigma_array_bin5_long[i] = (np.diag(np.matrix(Fmatdelensyslong).I))**0.5
        delenmid = np.delete(Fmatdelensyslong,(1,2),0)
        Fmatdelenlong = np.delete(delenmid,(1,2),1)
        delen_sigma_array_bin5_long[i] = (np.diag(np.matrix(Fmatdelenlong).I))**0.5
        delen_sigma_array_bin5_long_noncov[i] = 1/np.diag(Fmatdelenlong)**0.5
        Fmatdelensys1 = Fmatdelensyslong.copy()
        Fmatdelensys2 = Fmatdelensyslong.copy()
        Fmatdelensys1 = tools.add_prior(Fmatdelensys1, delensysparam_list, prior_dic1)
        Fmatdelensys2 = tools.add_prior(Fmatdelensys2, delensysparam_list, prior_dic2)
        delensys_sigma_array_bin5_inv1[i] = (np.diag(np.matrix(Fmatdelensys1).I))**0.5
        delensys_sigma_array_bin5_inv2[i] = (np.diag(np.matrix(Fmatdelensys2).I))**0.5

A_phi_sys_value=1.0e-18
alpha_phi_sys_value=-2.

n0s = np.loadtxt('params/generate_n0s_iter1st_rmsT%s_fwhmm%s_dl5.dat'%(rms_map_T_list_long[i], 1.0))
nels = n0s[:,0]

delensyslong_sigma_array_bin5_inv_short = np.column_stack((delensys_sigma_array_bin5_long[:,0], delensys_sigma_array_bin5_long[:,3:]))
delensys1_sigma_array_bin5_inv_short = np.column_stack((delensys_sigma_array_bin5_inv1[:,0], delensys_sigma_array_bin5_inv1[:,3:]))
delensys2_sigma_array_bin5_inv_short = np.column_stack((delensys_sigma_array_bin5_inv2[:,0], delensys_sigma_array_bin5_inv2[:,3:]))
#delensyslong_sigma_array_bin5_inv_short_noncov = np.column_stack((delensys_sigma_array_bin5_long_noncov[:,0], delensys_sigma_array_bin5_long_noncov[:,3:]))


plt.figure(figsize = (12,12))
for i, item in enumerate(param_list):
    axi = plt.subplot(3,3,i+1)
    #axi.loglog(rms_map_T_list_long, len_sigma_array_bin5_inv[:,i],'blue', label = 'lensed')
    axi.loglog(rms_map_T_list_long, len_sigma_array_bin5_inv_noncov[:,i],'blue', linestyle = '-.',label = 'lensed noncov')
    #axi.loglog(rms_map_T_list, unlen_sigma_array_bin5_inv[:,i],'red', label = 'unlensed')
    axi.loglog(rms_map_T_list, unlen_sigma_array_bin5_inv_noncov[:,i],'red', linestyle = '-.',label = 'unlensed noncov')
    #axi.loglog(rms_map_T_list_long, delen_sigma_array_bin5_long[:,i],'orange', label = 'delensed')
    #axi.loglog(rms_map_T_list_long, delensyslong_sigma_array_bin5_inv_short[:,i],'green', label = 'delensedsys')
    #axi.loglog(rms_map_T_list_long, delensyslong_sigma_array_bin5_inv_short[:,i],'purple', label = 'delensedsyslong')
    axi.loglog(rms_map_T_list_long, delen_sigma_array_bin5_long_noncov[:,i],'orange', label = 'delensed noncov', linestyle = '-.',) #'nomarginalization'
    #axi.loglog(rms_map_T_list_long, delensys1_sigma_array_bin5_inv_short[:,i],'green', linestyle = '--',label = '$\gamma_{\phi}$ prior 1')
    #axi.loglog(rms_map_T_list_long, delensys2_sigma_array_bin5_inv_short[:,i],'green', linestyle = '-.',label = '$\gamma_{N0}$ prior 1')
    #axi.loglog(rms_map_T_list, delensys_sigma_array_bin5_inv_short3[:,i],'green', linestyle = ':',label = 'gamma_prior=0.001')
    #axi.loglog(rms_map_T_list, delensys_sigma_array_bin5_inv_short[:,i]/unlen_sigma_array_bin5_inv[:,i],'orange')

    axi.legend()
    #axi.set_xlim(0,1)
    #axi.legend(['Bin5Len','Bin5Unlen','Bin5Delen','Bin5Delensys','Bin5Delensysbeta'],loc="lower center")

    axi.set_title(param_list[i])

axi = plt.subplot(3,3, i+2)
axi.loglog(rms_map_T_list_long, delensys_sigma_array_bin5_long[:,1],'purple', label='$\gamma_{N}$')
axi.loglog(rms_map_T_list_long, delensys_sigma_array_bin5_long[:,2],'green', label='$\gamma_{\phi}$')
axi.legend()
#axi.legend(['Bin5delensys'],loc="lower center")
axi.set_title("gamma")


plt.tight_layout()
plt.savefig('ForNote/compare_2gammas1.00_iter1st_addlensiing_dl%s_%s_rem6_%s2.png'%(dl, camborself, systype))

i = 0
unlencovname = "diff_covariance/new_delta_dict_%s_n%s_addlensing.npy"%("unlensed_total", rms_map_T_list[3])
lencovname = "diff_covariance/new_delta_dict_%s_n%s_addlensing.npy"%("total", rms_map_T_list[3])
delencovname = "diff_covariance/new_delta_dict_%s_%s%s_n%s_addlensing.npy"%('iter1st',"delensed_scalar",sysornot, rms_map_T_list[3])

len_delta_dict = np.load(lencovname)
len_delta_dict = len_delta_dict.item()

unlen_delta_dict = np.load(unlencovname)
unlen_delta_dict = unlen_delta_dict.item()

delen_delta_dict = np.load(delencovname)
delen_delta_dict = delen_delta_dict.item()

clnames = ['TT','EE','TE','BB','PP']
plt.clf()
pnum = len(clnames)
pnum2 = len(param_list)
fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum, nrows=pnum)
#covmat = np.block([[new_delta_dict['TTTT'], new_delta_dict['TTEE'], new_delta_dict['TTTE'],new_delta_dict['BBTT'].T,new_delta_dict['PPTT'].T],[new_delta_dict['TTEE'].T, new_delta_dict['EEEE'], new_delta_dict['EETE'], new_delta_dict['BBEE'].T,new_delta_dict['PPEE'].T],[new_delta_dict['TTTE'].T, new_delta_dict['EETE'], new_delta_dict['TETE'], new_delta_dict['BBTE'].T,new_delta_dict['PPTE'].T],[new_delta_dict['BBTT'], new_delta_dict['BBEE'], new_delta_dict['BBTE'], new_delta_dict['BBBB'], new_delta_dict['PPBB'].T],[new_delta_dict['PPTT'], new_delta_dict['PPEE'],new_delta_dict['PPTE'],new_delta_dict['PPBB'],new_delta_dict['PPPP']]])
#covmat = np.asarray(covmat)

fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum, nrows=pnum)

for i, p1 in enumerate(clnames):
    for j, p2 in enumerate(clnames):
        print(p1,p2)
        if i<j:
            axes[i,j].axis('off')
        else:
            if p1 == 'TT' or p1 == 'EE' or p1 == 'TE':
                covmat = len_delta_dict[p2+p1]
            elif p1 == 'BB' or p1 == 'PP':
                covmat = len_delta_dict[p1+p2]
            sigma1 = np.diag(len_delta_dict[p1+p1])
            sigma2 = np.diag(len_delta_dict[p2+p2])
            norm = np.einsum('i,j->ij',sigma1, sigma2)
            corremat = covmat/norm**0.5
            zerodiag = np.ones(corremat.shape)
            np.fill_diagonal(zerodiag, 0)
            im = axes[i,j].imshow(corremat*zerodiag, vmin = -0.002, vmax = 0.002)
            #plt.colorbar()
            if j == 0:
                axes[i,j].set_ylabel(p1)
            if i == pnum-1:
                axes[i,j].set_xlabel(p2)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
#fig.colorbar(im)
#fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig('ForNote/covmat_addlensing_len1uk.png')

fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum, nrows=pnum)

for i, p1 in enumerate(clnames):
    for j, p2 in enumerate(clnames):
        if i<j:
            axes[i,j].axis('off')
        else:
            if p1 == 'TT' or p1 == 'EE' or p1 == 'TE':
                covmat = len_delta_dict[p2+p1]
            elif p1 == 'BB' or p1 == 'PP':
                covmat = len_delta_dict[p1+p2]
            sigma1 = np.diag(delen_delta_dict[p1+p1])
            sigma2 = np.diag(delen_delta_dict[p2+p2])
            norm = np.einsum('i,j->ij',sigma1, sigma2)
            corremat = covmat/norm**0.5
            zerodiag = np.ones(corremat.shape)
            np.fill_diagonal(zerodiag, 0)
            axes[i,j].imshow(corremat*zerodiag)
            #axes[i,j].set_colorbar()
            if j == 0:
                axes[i,j].set_ylabel(p1)
            if i == pnum-1:
                axes[i,j].set_xlabel(p2)


plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('ForNote/covmat_addlensing_delen1uk.png')
#'''



plt.clf()
#cb=plt.colorbar()
fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum, nrows=pnum)
#cb=plt.colorbar()

def updatecov(ni):
    global im, cb
    im.remove()
    #plt.clf()
    #fig.clear()
    #cb=plt.colorbar()
    #cb.remove()
    try:
        cb
    except NameError:
        cb_exists = False
    else:
        cb_exists = True

    if cb_exists:
        cb.remove()  
    
    lencovname = "diff_covariance/new_delta_dict_%s_n%s_addlensing.npy"%("total", ni)
    len_delta_dict = np.load(lencovname)
    len_delta_dict = len_delta_dict.item()

    for i, p1 in enumerate(clnames):
        for j, p2 in enumerate(clnames):
            if i<j:
                axes[i,j].axis('off')
            else:
                if p1 == 'TT' or p1 == 'EE' or p1 == 'TE':
                    covmat = len_delta_dict[p2+p1]
                elif p1 == 'BB' or p1 == 'PP':
                    covmat = len_delta_dict[p1+p2]
                sigma1 = np.diag(len_delta_dict[p1+p1])
                sigma2 = np.diag(len_delta_dict[p2+p2])
                norm = np.einsum('i,j->ij',sigma1, sigma2)
                corremat = covmat/norm**0.5
                zerodiag = np.ones(corremat.shape)
                np.fill_diagonal(zerodiag, 0)
                im = axes[i,j].imshow(corremat*zerodiag, vmin = -0.002, vmax = 0.006)
                #plt.colorbar()
                if j == 0:
                    axes[i,j].set_ylabel(p1)
                if i == pnum-1:
                    axes[i,j].set_xlabel(p2)

    plt.subplots_adjust(wspace=0, hspace=0)
    cax,kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
    cb = plt.colorbar(im, cax=cax, **kw)

    #fig.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    fig.suptitle('n = %suk'%(ni), fontsize = 15)
    return fig,

ani = FuncAnimation(fig, updatecov, rms_map_T_list, interval=500, blit=False)
ani.save('covmatlen2.gif')    
#'''


Fmatlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_addlensing.npy"%('zeroN0',"total", systype, rms_map_T_list[0]))
Fmatunlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_addlensing.npy"%('zeroN0',"unlensed_total", systype, rms_map_T_list[0]))
Fmatdelensys = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_addlensing.npy"%('iter1st',"delensed_scalar", systype, rms_map_T_list[0]))

Fmatdelen = np.delete(Fmatdelensys, (1,2), axis = 0)
Fmatdelen = np.delete(Fmatdelen, (1,2), axis = 1)

cov_matdelensys0 = np.matrix(Fmatdelensys).I
cov_matdelen0 = np.matrix(Fmatdelen).I
cov_matlen0 = np.matrix(Fmatlen).I
cov_matunlen0 = np.matrix(Fmatunlen).I

plt.clf()
fig, axes = plt.subplots(figsize=(16, 16), ncols=pnum2, nrows=pnum2)

def update2d(ni):
    Fmatlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_addlensing.npy"%('zeroN0',"total", systype, ni))
    Fmatunlen = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_addlensing.npy"%('zeroN0',"unlensed_total", systype, ni))
    Fmatdelensys = np.load("ForNote/F_nongau_mat_%s_%s%s_n%s_addlensing.npy"%('iter1st',"delensed_scalar", systype, ni))

    Fmatdelen = np.delete(Fmatdelensys, (1,2), axis = 0)
    Fmatdelen = np.delete(Fmatdelen, (1,2), axis = 1)

    cov_matdelensys = np.matrix(Fmatdelensys).I
    cov_matdelensys = np.delete(cov_matdelensys, (1,2), axis = 0)
    cov_matdelensys = np.delete(cov_matdelensys, (1,2), axis = 1)
    cov_matdelen = np.matrix(Fmatdelen).I
    cov_matlen = np.matrix(Fmatlen).I
    cov_matunlen = np.matrix(Fmatunlen).I
    '''
    global cssys
    global cs
    global cslen
    global csunlen

    cssys.remove()
    cs.remove()
    cslen.remove()
    csunlen.remove()
    '''
    for i, p1 in enumerate(param_list):
        for j, p2 in enumerate(param_list):
            if i<j:
                axes[i,j].axis('off')
                #for j, p2 in enumerate(param_list[:i+1]):        
            elif i==j:
                sigmai = cov_matdelen[i,i]**0.5
                sigmai0 = cov_matdelen0[i,i]**0.5
                sigmasysi = cov_matdelensys[i,i]**0.5
                sigmailen = cov_matlen[i,i]**0.5
                sigmaiunlen = cov_matunlen[i,i]**0.5
                valuei = para_value[p1]
                xst = valuei - sigmai0*5
                xed = valuei + sigmai0*5
                x = np.linspace(xst, xed, 240)            
                coe = 1.0 / (2 * np.pi)**0.5 / sigmai
                Z = np.exp (-0.5 * ((x-valuei)/sigmai)**2)
                Zsys = np.exp (-0.5 * ((x-valuei)/sigmasysi)**2)
                Zlen = np.exp (-0.5 * ((x-valuei)/sigmailen)**2)
                Zunlen = np.exp (-0.5 * ((x-valuei)/sigmaiunlen)**2)
                axes[i,i].plot(x, Zsys, 'yellowgreen', label = 'delensedsys')
                axes[i,i].plot(x, Z, 'olive', label = 'delensed')
                axes[i,i].plot(x, Zlen, 'blue', label = 'lensed')
                axes[i,i].plot(x, Zunlen, 'orange', label = 'unlensed')
                #axes[i,i].legend()
                if i == 0:
                    axes[i,j].set_ylabel(p1)

                    #axes[i,j].xlim
            else:
                sigmai = cov_matdelen[i,i]**0.5
                sigmaj = cov_matdelen[j,j]**0.5
                sigmai0 = cov_matdelen0[i,i]**0.5
                sigmaj0 = cov_matdelen0[j,j]**0.5
                sigmasysi = cov_matdelensys[i,i]**0.5
                sigmasysj = cov_matdelensys[j,j]**0.5
                valuei = para_value[p1]
                valuej = para_value[p2]
                sigmailen = cov_matlen[i,i]**0.5
                sigmajlen = cov_matlen[j,j]**0.5
                sigmaiunlen = cov_matunlen[i,i]**0.5
                sigmajunlen = cov_matunlen[j,j]**0.5
                xst = valuej - sigmaj0*5
                xed = valuej + sigmaj0*5
                yst = valuei - sigmai0*5
                yed = valuei + sigmai0*5
                x = np.linspace(xst, xed, 280)
                y = np.linspace(yst, yed, 280)
                X,Y = np.meshgrid(x,y)
                coe = 1.0 / ((2 * np.pi)**2 * np.linalg.det(cov_matlen))**0.5
                Fmatdelen = np.matrix([[cov_matdelen[j,j], cov_matdelen[j,i]],[cov_matdelen[i,j],cov_matdelen[i,i]]]).I
                Fmatdelensys = np.matrix([[cov_matdelensys[j,j], cov_matdelensys[j,i]],[cov_matdelensys[i,j],cov_matdelensys[i,i]]]).I
                Fmatlen = np.matrix([[cov_matlen[j,j], cov_matlen[j,i]],[cov_matlen[i,j],cov_matlen[i,i]]]).I
                Fmatunlen = np.matrix([[cov_matunlen[j,j], cov_matunlen[j,i]],[cov_matunlen[i,j],cov_matunlen[i,i]]]).I


                Z = np.exp(-0.5 * (Fmatdelen[0,0]*(X-valuej)**2 + (Fmatdelen[0,1] + Fmatdelen[1,0])*(X-valuej)*(Y-valuei) + Fmatdelen[1,1]*(Y-valuei)**2))
                Zsys = np.exp(-0.5 * (Fmatdelensys[0,0]*(X-valuej)**2 + (Fmatdelensys[0,1] + Fmatdelensys[1,0])*(X-valuej)*(Y-valuei) + Fmatdelensys[1,1]*(Y-valuei)**2))
                z1 = np.exp(-chi2.ppf(0.683, df=2)/2)
                z3 = np.exp(-chi2.ppf(0.997, df=2)/2)

                Zlen = np.exp(-0.5 * (Fmatlen[0,0]*(X-valuej)**2 + (Fmatlen[0,1] + Fmatlen[1,0])*(X-valuej)*(Y-valuei) + Fmatlen[1,1]*(Y-valuei)**2))
                Zunlen = np.exp(-0.5 * (Fmatunlen[0,0]*(X-valuej)**2 + (Fmatunlen[0,1] + Fmatunlen[1,0])*(X-valuej)*(Y-valuei) + Fmatunlen[1,1]*(Y-valuei)**2))

                cssys = axes[i,j].contourf(X,Y,Zsys,[z1, 1], colors=['yellowgreen', 'honeydew'])
                cs = axes[i,j].contourf(X,Y,Z,[z1, 1], colors=['olive', 'orange'])
                cslen = axes[i,j].contour(X,Y,Zlen,[z1, 1], colors=['blue', 'yellow'])
                csunlen = axes[i,j].contour(X,Y,Zunlen,[z1, 1], colors=['orange', 'yellow'])
                axes[i,j].set_ylim([yst, yed])
                axes[i,j].set_xlim([xst, xed])

                if j == 0:
                    axes[i,j].set_ylabel(p1)
                if i == pnum2-1:
                    axes[i,j].set_xlabel(p2)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()
    return axes

ani = FuncAnimation(fig, update2d, rms_map_T_list, interval=500, blit=False)
ani.save('ForNote/fisher2d_sys_addlensing.gif')    
