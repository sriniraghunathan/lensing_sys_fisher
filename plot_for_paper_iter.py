import numpy as np, scipy as sc, sys, argparse, os
import tools
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy import interpolate 
from scipy.interpolate import interp1d
from os.path import exists
import json
import copy
import camb
from datetime import datetime
import re

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

############################################################################################################
#get the necessary arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt')
#parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='delensed_scalar', choices=['delensed_scalar', 'unlensed_total', 'total']) # add delensed
parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='delensed_scalar', choices=['delensed_scalar', 'unlensed_total', 'total']) # add delensed

#reduce lensing amplitude by xx per cent. Roughly mimicking S4-Wide delensing.
parser.add_argument('-Alens', dest='Alens', action='store', help='Alens', type=float, default=1) 
#parser.add_argument('-Alens', dest='Alens', action='store', help='Alens', type=float, default=0.3) 

parser.add_argument('-fsky', dest='fsky', action='store', help='fsky', type = float, default = 0.57)
parser.add_argument('-include_lensing', dest='include_lensing', action='store', help='include_lensing', type = int, default = 0)
#xx percent of lensing N0 will be considered as lensing systematic
parser.add_argument('-lensing_sys_n0_frac', dest='lensing_sys_n0_frac', action='store', help='lensing_sys_n0_frac', type = float, default = 0.2)

#ILC residual file
parser.add_argument('-use_ilc_nl', dest='use_ilc_nl', action='store', help='use_ilc_nl', type=int, default = 0)
#or add noise levels (only used if use_ilc_nl = 0)
parser.add_argument('-rms_map_T', dest='rms_map_T', action='store', help='rms_map_T', type=float, default = -1)
parser.add_argument('-fwhm_arcmins', dest='fwhm_arcmins', action='store', help='fwhm_arcmins', type=float, default = 1.4)

#debug
parser.add_argument('-debug', dest='debug', action='store', help='debug', type=int, default = 0)

args = parser.parse_args()
args_keys = args.__dict__

for kargs in args_keys:
    param_value = args_keys[kargs]
    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

if debug: from pylab import *
logline='\nstart Fisher forecasting\n'; tools.write_log(logline)

if rms_map_T == -1:
    rms_map_T_list = np.array((0.2, 0.5, 2))#np.arange(1,11,1)
    #rms_map_T_list = np.array((0.1, 0.2, 0.5, 1, 2))#np.arange(1,11,1)
    #rms_map_T_list = np.array((0.2, 0.5, 1, 2))#np.arange(1,11,1)
    #rms_map_T_list = np.array((0.1, 0.2, 1, 2))#np.arange(1,11,1)

else:
    rms_map_T_list = [rms_map_T]
fwhm_list = np.ones(len(rms_map_T_list))

############################################################################################################
#get experiment specs
logline = '\tset lmax / power spectra to be used'; tools.write_log(logline)
if include_lensing:
    pspectra_to_use = ['TT', 'EE', 'TE', 'PP'] #CMB TT/EE/TE/lensing
else:
    pspectra_to_use = ['TT', 'EE', 'TE'] #CMB TT/EE/TE
min_l_temp, max_l_temp = 30, 3000
min_l_pol, max_l_pol = 30, 5000
############################################################################################################

############################################################################################################
#cosmological parameters

params_to_constrain = ['As','tau','r','ns', 'ombh2', 'omch2', 'thetastar','neff','gamma_phi_sys','mnu']
fix_params = ['Alens', 'ws', 'omk']#, 'mnu'] #parameters to be fixed (if included in fisher forecasting)
#fix_params = ['r','ns', 'ombh2', 'thetastar']
#prior_dic = {'tau':0.007} #Planck-like tau prior
prior_dic = {}

if lensing_sys_n0_frac>0.:
    pass

desired_param_arr = None ##['ns', 'neff'] #desired parameter to be printed. Prints everything is set to None

#folders containing inputs
#camb_folder = 'data/CMB_spectra_derivatives_for_code_comparison/'

if use_ilc_nl:
    draft_results_folder = 'data/DRAFT_results_20200601/s4like_mask/TT-EE-TE/baseline/'
else:
    draft_results_folder = None
############################################################################################################

############################################################################################################
#get/read the parameter file
logline = '\tget/read the parameter file'; tools.write_log(logline)
param_dict = tools.get_ini_param_dict(paramfile)
print('The parameters used are:\n')
print(param_dict)
param_dict['Alens'] = Alens
systype = 'changefield'
systype = 'guessgamma'
systype = 'gphionly'
systype = 'gphi4all'

Lsdl = 5
binsize = 5   #rebin in calculating the cov^-1
param_dict['binsize'] = binsize
param_dict['Lsdl'] = Lsdl
#iteration = True
iteration = False
addlensing = False
#addlensing = True

addBB = True
#addBB = False

if addlensing == False and addBB == False:
    tail = 'TTEETE'
if addlensing == True and addBB == True:
    tail = 'addlensing'
elif addlensing == False and addBB == True:
    tail = 'noPhi'
elif addBB == False and addlensing == True:
    tail = 'noBB'

itername = ''
if iteration == True:
    itername = 'iter1st'
else:
    itername = 'iter0'
'''
else:
    itername = 'zeroN0'
'''

############################################################################################################


############################################################################################################

nl_dict = {}

els, powerini = tools.get_ini_cmb_power(param_dict, raw_cl = 1)
unlensedCL = powerini['unlensed_total']
totCL = powerini['total']
cl_phiphi, cl_Tphi, cl_Ephi = powerini['lens_potential'].T
cphifun = interpolate.interp1d(els, cl_phiphi)

Bl_list = [0]*len(fwhm_list)
nl_TT_list = [0]*len(fwhm_list)
nl_PP_list = [0]*len(fwhm_list)

for i, item in enumerate(rms_map_T_list):
    Bl_list[i], nl_TT_list[i], nl_PP_list[i] = tools.get_nl(els, rms_map_T = item, rms_map_P = item*2.**0.5, fwhm = fwhm_list[i])

use_foreground = False
sample_var_percent = 0.1
dustdict = {}

if use_ilc_nl:
    include_gal = 1
    gal_mask = 3 #only valid if galaxy is included
    if not include_gal:
        nlfile = '%s/S4_ilc_galaxy0_27-39-93-145-225-278_TT-EE-TE_lf2-mf12-hf5_AZ.npy' %(draft_results_folder)
    else:
        nlfile = '%s/S4_ilc_galaxy1_27-39-93-145-225-278_TT-EE-TE_lf2-mf12-hf5_galmask%s_AZ.npy' %(draft_results_folder, gal_mask)
    nl_dict = tools.get_nl_dict(nlfile, els)
    nl_TT = nl_dict['TT']
    nl_PP = nl_dict['EE']


dust150dict = {}

noise_fname0 = 'params/generate_n0s_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
#noise_fname0_new = 'params/generate_n0EBs_un_plus_n_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
noise_fname_iter1st = 'params/generate_n0s_EBiter1st_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
#noise_fname_iter1st_new = 'params/generate_n0EBs_un_plus_n_iter1st_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
#output_name = "params/generate_n0EBs_un_plus_n_iter1stdiff_rmsT%s_fwhm%s_dl5.dat"%(rms_map_T_list[i], 1.0)

gamma_phi_sys_guess = 1
gamma_phi_sys_value = 0.80

delenBB_dict = {}
delenBBiter_dict = {}
cres_dict = {}
cresiter_dict = {}
N0_dict = {}
origN0_dict = {}

#filename = "delensedpoweriter_new_non.npy"
filename = "delensedpoweriter_new_intep_gamma%s.npy"%(gamma_phi_sys_value)
file_exists = exists(filename)

if not file_exists:
    for i in range(len(rms_map_T_list)):
        ni = rms_map_T_list[i]
        #n0filename = 'params/generate_n0EBs_un_plus_n_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
        #n1filename = 'params/generate_n0EBs_un_plus_n_iter1st_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
        #nxfilename = 'params/generate_n0EBs_un_plus_n_iter1stdiff_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0        
        n0filename = 'params/generate_n0s_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
        n1filename = 'params/generate_n0s_EBiter1st_rmsT%s_fwhm%s_dl5.dat'%(rms_map_T_list[i], 1.0)

        file_exists = exists(n0filename)
        print('Calculating noise%s'%(rms_map_T_list[i]))

        n0s = np.loadtxt(n0filename)
        n1s = np.loadtxt(n1filename)
        #nxs = np.loadtxt(nxfilename)

        eb0 = n0s[:,1]
        eb1 = n1s[:,1]
        #ebx = nxs[:,1]
        nels = n0s[:,0]
        nels1 = n1s[:,0]
        N0dict  ={}
        N0dict['N0'] = eb0
        N0dict['nels'] = nels

        #Nxdict  ={}
        #Nxdict['N0'] = nxs[:,2]

        N1dict  ={}
        N1dict['N0'] = n1s[:,1]
        
        dl2els = ((1+els)*els)**2/2/np.pi

        dl2 = ((1+nels)*nels)**2/2/np.pi
        n0fun = interpolate.interp1d(nels, eb0*dl2,kind='linear')
        n0els = n0fun(els)/dl2els

        dl2 = ((1+nels1)*nels1)**2/2/np.pi
        n1fun = interpolate.interp1d(nels1, eb1*dl2, kind = 'linear')
        n1els = n1fun(els)/dl2els

        #nxfun = interpolate.interp1d(nels, ebx)
        #nxels = nxfun(els)

        N0_dict[ni] = {}
        origN0_dict[ni] = {}
        origN0_dict[ni]['iter0'] = eb0
        origN0_dict[ni]['nels'] = nels
        N0_dict[ni]['iter0'] = n0els
        N0_dict[ni]['iter1'] = n1els
        print("nels shape", nels.shape)
        #N0_dict[ni]['cross'] = ebx
        ############################################################################################################         
        #get delensed power

        cphi_tot_value = n0els + cl_phiphi*gamma_phi_sys_value**2
        winf = gamma_phi_sys_guess*cl_phiphi / cphi_tot_value 
        clpp0 = (winf**2 * cphi_tot_value - 2*gamma_phi_sys_value*winf * cl_phiphi + cl_phiphi) * (els*(els+1))**2/2/np.pi
        clppguess0 = (winf**2 * cphi_tot_value - 2*gamma_phi_sys_guess*winf * cl_phiphi + cl_phiphi) * (els*(els+1))**2/2/np.pi

        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        clpp = np.insert(clpp0, 0, np.zeros(2), axis = 0)
        clppguess = np.insert(clppguess0, 0, np.zeros(2), axis = 0)
        thyres = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        thyresguess = camb.correlations.lensed_cls(cls, clppguess, lmax = els[-1])

        powers = {}
        powers['value'] = thyres[param_dict['min_l_limit']:, :]
        powers['value'] = powers['value'] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
        powers["guess"] = thyresguess[param_dict['min_l_limit']:, :]
        powers["guess"] = powers["guess"] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
        cl_tt, cl_ee, cl_bb, cl_te = powers['value'].T
        delenBB_dict[ni] = cl_bb
        cl_tt_guess, cl_ee_guess, cl_bb_guess, cl_te_guess = powers["guess"].T
        delenBB_dict["%sguess"%(ni)] = cl_bb_guess
        cres_dict[ni] = {}
        cres_dict[ni]['value'] = clpp0
        cres_dict[ni]['guess'] = clppguess0

        # for iter
        cphi_tot_value = n0els + cl_phiphi*gamma_phi_sys_value**2
        winf = gamma_phi_sys_guess*cl_phiphi / cphi_tot_value 
        clpp0 = (winf**2 * cphi_tot_value - 2*gamma_phi_sys_value*winf * cl_phiphi + cl_phiphi) * (els*(els+1))**2/2/np.pi
        winfiter = gamma_phi_sys_guess**2*clpp0 / (clpp0*gamma_phi_sys_value**2 + n1els*(els*(els+1))**2/2/np.pi )
        #clpp0iter = ((1-gamma_phi_sys_value**2*winfiter)**2*clpp0 + winfiter**2*n1els - 2*winfiter*(1-winfiter)*winf*nxels)* (els*(els+1))**2/2/np.pi
        clpp0iter = (1-gamma_phi_sys_value**2*winfiter)**2*clpp0 + winfiter**2*n1els* (els*(els+1))**2/2/np.pi
        #clppguess0iter = ((1-gamma_phi_sys_guess**2*winfiter)**2*clppguess0 + winfiter**2*n1els - 2*winfiter*(1-winfiter)*winf*nxels)* (els*(els+1))**2/2/np.pi
        clppguess0iter = (1-gamma_phi_sys_guess**2*winfiter)**2*clppguess0 + winfiter**2*n1els* (els*(els+1))**2/2/np.pi

        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)

        clppiter = np.insert(clpp0iter, 0, np.zeros(2), axis = 0)
        clppguessiter = np.insert(clppguess0iter, 0, np.zeros(2), axis = 0)
        thyres = camb.correlations.lensed_cls(cls, clppiter, lmax = els[-1])
        #thyres = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        thyresguess = camb.correlations.lensed_cls(cls, clppguessiter, lmax = els[-1])

        powersiter = {}
        powersiter['value'] = thyres[param_dict['min_l_limit']:, :]
        powersiter['value'] = powersiter['value'] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
        powersiter["guess"] = thyresguess[param_dict['min_l_limit']:, :]
        powersiter["guess"] = powersiter["guess"] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
        cl_tt, cl_ee, cl_bb, cl_te = powersiter['value'].T
        delenBBiter_dict[ni] = cl_bb
        cl_tt_guess, cl_ee_guess, cl_bb_guess, cl_te_guess = powersiter["guess"].T
        delenBBiter_dict["%sguess"%(ni)] = cl_bb_guess
        cresiter_dict[ni] = {}
        cresiter_dict[ni]['value'] = clpp0iter
        cresiter_dict[ni]['guess'] = clppguess0iter
        
    np.save("cres_new_intep_gamma%s.npy"%(gamma_phi_sys_value), cres_dict)
    np.save("cresiter_new_intep_gamma%s.npy"%(gamma_phi_sys_value), cresiter_dict)
    #np.save("cres_new_intep.npy", cres_dict)
    #np.save("cresiter_new_intep.npy", cresiter_dict)
    np.save("delensedpower_new_intep_gamma%s.npy"%(gamma_phi_sys_value), delenBB_dict)
    np.save("delensedpoweriter_new_intep_gamma%s.npy"%(gamma_phi_sys_value), delenBBiter_dict)
    #np.save("N0EB_dict_intep.npy", N0_dict)
    #np.save("origN0EB_dict.npy", origN0_dict)
    print("please run it again to finish the plot")

else:
    delenBBsys_dict = np.load("delensedpower_new_intep_gamma%s.npy"%(gamma_phi_sys_value))
    delenBB_dict = np.load("delensedpower_new_intep.npy")
    #filename = "delensedpoweriter_gamma0.98.npy"

    delenBBitersys_dict = np.load("delensedpoweriter_new_intep_gamma%s.npy"%(gamma_phi_sys_value))
    delenBBiter_dict = np.load("delensedpoweriter_new_intep.npy")
    delenBBsys_dict = delenBBsys_dict.item()
    delenBB_dict = delenBB_dict.item()
    delenBBitersys_dict = delenBBitersys_dict.item()
    delenBBiter_dict = delenBBiter_dict.item()
    cres_dict = np.load("cres_new_intep_gamma%s.npy"%(gamma_phi_sys_value))
    cresiter_dict = np.load("cresiter_new_intep_gamma%s.npy"%(gamma_phi_sys_value))
    cres_dict = cres_dict.item()
    cresiter_dict = cresiter_dict.item()
    N0_dict = np.load("N0EB_dict_intep.npy")
    N0_dict = N0_dict.item()
    origN0_dict = np.load("origN0EB_dict.npy")
    origN0_dict = origN0_dict.item()
#######################################
#delensed BB

plt.clf()
dl = els*(els+1)/2/np.pi
plt.loglog(els, dl*unlensedCL[:,2], label = 'unlensed')
plt.loglog(els, dl*totCL[:,2], label = 'lensed')
for i, ni in enumerate(rms_map_T_list):
    print(i, ni)
    lines = plt.loglog(els, dl*delenBBsys_dict[ni], label = "delen sys n %suk-arcmin"%(ni))
    color = lines[0].get_color()
    plt.loglog(els, dl*delenBBitersys_dict[ni], label = "delen iter sys n %suk-arcmin"%(ni), color = color, linestyle = "--")
    plt.loglog(els, dl*delenBB_dict[ni], label = "delen n %suk-arcmin"%(ni), color = color, linestyle = "-.")
    plt.loglog(els, dl*delenBBiter_dict[ni], label = "delen iter n %suk-arcmin"%(ni), color = color, linestyle = ":")
    #plt.loglog(els, dl*nl_TT_list[i], label = "white noise = %suk-arcmin"%(ni), color = color, linestyle = ":")
plt.legend(loc='lower right')
plt.xlabel("$\ell$")
plt.xlim(2, 2000)
plt.ylabel("$\ell(\ell + 1)C_{\ell}^{BB}/2\pi[\mu k^2]$")
plt.savefig("delensedBBiter_noiseBB_gamma%s.png"%(gamma_phi_sys_value))


plt.clf()
dl = els*(els+1)/2/np.pi
for ni in rms_map_T_list:
    eff = 1-(delenBB_dict[ni] - totCL[:,2]) / (unlensedCL[:,2] - totCL[:,2])
    effiter = 1- (delenBBiter_dict[ni] - totCL[:,2]) / (unlensedCL[:,2] - totCL[:,2])
    effsys = 1-(delenBBsys_dict[ni] - totCL[:,2]) / (unlensedCL[:,2] - totCL[:,2])
    effitersys = 1- (delenBBitersys_dict[ni] - totCL[:,2]) / (unlensedCL[:,2] - totCL[:,2])
    lines = plt.plot(els, effsys, label = "delen sys n%suk-arcmin"%(ni))
    color = lines[0].get_color()
    plt.plot(els, effitersys, label = "iterdelen sys n%suk-arcmin"%(ni), color = color, linestyle = '--')
    plt.plot(els, eff, label = "delen n%suk-arcmin"%(ni), color = color, linestyle = '-.')
    plt.plot(els, effiter, label = "iterdelen n%suk-arcmin"%(ni), color = color, linestyle = ':')
plt.legend()
plt.xlabel("$\ell$")
plt.ylabel("$ratio$")
plt.savefig("delensedBBiter_res_gamma%s.png"%(gamma_phi_sys_value))

################################################
#cres

plt.clf()
dl2 = (els*(els+1))**2/2/np.pi
plt.loglog(els, dl2*cl_phiphi, label = 'unlensed', color = 'black')
for i, ni in enumerate(rms_map_T_list):
    nels = origN0_dict[ni]['nels']
    dl3 = (nels*(nels+1))**2/2/np.pi
    #lines = plt.loglog(nels, dl3*origN0_dict[ni]['iter0'], label = "orig n0 = %suk-arcmin"%(ni))
    lines = plt.loglog(els, cres_dict[ni]['value'], label = "cres value = %suk-arcmin"%(ni))
    color = lines[0].get_color()
    plt.loglog(els, cresiter_dict[ni]['value'], label = "cresiter = %suk-arcmin"%(ni), color = color, linestyle = "--")
    #plt.loglog(els, dl2*N0_dict['n0'], label = "cresiter = %suk-arcmin"%(ni), linestyle = "")
    #color = lines[0].get_color()
    plt.loglog(els, dl2*N0_dict[ni]['iter0'], label = "n0 = %suk-arcmin"%(ni), color = color, linestyle = "-.")
    plt.loglog(els, dl2*N0_dict[ni]['iter1'], label = "n1 = %suk-arcmin"%(ni), color = color, linestyle = ":")
    #plt.loglog(els, dl*nl_TT_list[i], label = "white noise = %suk-arcmin"%(ni), color = color, linestyle = ":")
plt.legend(loc='lower right')
plt.xlabel("$\ell$")
plt.xlim(2, 2000)
plt.ylabel("$(\ell(\ell + 1))^2C_{\ell}^{\phi\phi}/2\pi[\mu k^2]$")
plt.savefig("cres_new_intp_gamma%s.png"%(gamma_phi_sys_value))

