"""
#CMB fisher forecasting
"""
############################################################################################################

import numpy as np, scipy as sc, sys, argparse, os
import tools
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy import interpolate 
from scipy.interpolate import interp1d
from os.path import exists
import json

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
parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='total', choices=['delensed_scalar', 'unlensed_total', 'total']) # add delensed

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
    rms_map_T_list = np.array((0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10))#np.arange(1,11,1)
    rms_map_T_list = np.array((0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10))#np.arange(1,11,1)
    rms_map_T_list = np.array((0.1, 0.2, 0.5, 1, 2, 5, 10))#np.arange(1,11,1)
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
#params_to_constrain = ['As','tau','r','ns', 'ombh2', 'omch2', 'thetastar','neff','gamma_phi_sys', 'mnu']
params_to_constrain = ['As','tau','ns', 'ombh2', 'omch2', 'thetastar','neff', 'mnu']
params_to_constrain = ['As','tau','ns', 'ombh2', 'omch2', 'thetastar','neff', 'mnu', 'r', 'gamma_phi_sys']
params_to_constrain = ['As','tau','ns', 'ombh2', 'omch2', 'thetastar','neff', 'mnu', 'r']
fix_params = ['Alens', 'ws', 'omk']#, 'mnu'] #parameters to be fixed (if included in fisher forecasting)
#fix_params = ['r','ns', 'ombh2', 'thetastar']
#prior_dic = {'tau':0.007} #Planck-like tau prior
#prior_dic = {'tau':0.002} #Planck-like tau prior
prior_dic = {}
# prior 1: A:1e-17, alpha:1, prior2: A:5e-18, alpha:1, prior3: A:5e-18, alpha:1,
if lensing_sys_n0_frac>0.:
    pass
    #prior_dic = {'Asys_lens':0.1, 'alphasys_lens': 0.1} #play with the choice of prior here.
desired_param_arr = None ##['ns', 'neff'] #desired parameter to be printed. Prints everything is set to None
############################################################################################################

############################################################################################################
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

Lsdl = 5
binsize = 5   #rebin in calculating the cov^-1
param_dict['binsize'] = binsize
param_dict['Lsdl'] = Lsdl
#iteration = True
iteration = False
addlensing = False
#addlensing = True
itername = ''

if which_spectra == 'delensed_scalar':
    print('here!')
    if iteration == True:
        itername = 'iter1st'
    else:
        itername = 'iter0'
else:
    itername = 'zeroN0'


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

#exit()
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


nl_dict = {}

for i in range(len(rms_map_T_list)):
    if i > 10:
        continue
    n0filename = 'params/generate_n0s_iter1st_rmsT%s_fwhmm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
    #n0filename = 'params/generate_n0s_rmsT%s_fwhmm%s_dl5.dat'%(rms_map_T_list[i], 1.0)
    file_exists = exists(n0filename)
    print('Calculating noise%s'%(rms_map_T_list[i]))

    if file_exists:
        print("Already have N0 for this noise level!!! \n" )
        n0s = np.loadtxt(n0filename)
        mv = n0s[:,1]
        nels = n0s[:,0]
        nl_mv = interpolate.interp1d(nels, mv)
        nl_dict['PP'] = nl_mv(els)
        param_dict['rms_map_T'] = rms_map_T_list[i]
        param_dict['rms_map_P'] = rms_map_T_list[i] * 2**0.5
        param_dict['nlP'] = nl_PP_list[i]
        param_dict['nlT'] = nl_TT_list[i]
        param_dict['fwhm_arcmins'] = fwhm_list[i]
    else:
        print("The iteration N0 doesn't exist for this noise level!")
        continue

    nl_dict['TT'] = nl_TT_list[i]
    nl_dict['EE'] = nl_PP_list[i]
    nl_dict['BB'] = nl_PP_list[i]
    nl_dict['TE'] = nl_TT_list[i]*0
    nl_dict['SYS'] = tools.get_nl_sys(els, param_dict['A_phi_sys'], param_dict['alpha_phi_sys'])
    nl_dict['CROSS'] = param_dict['beta_phi_sys'] * (cl_phiphi*nl_dict['SYS'])**0.5
    

    ############################################################################################################
    #get fiducual LCDM power spectra computed using CAMB
    logline = '\tget fiducual LCDM %s power spectra computed using CAMB' %(which_spectra); tools.write_log(logline)
    pars, els, cl_dict = tools.get_cmb_spectra_using_camb(param_dict, which_spectra, noise_nzero_fname = n0filename)
    ############################################################################################################

    ############################################################################################################
    #get derivatives
    sysornot = 'sys'
    logline = '\tget/read derivatives'; tools.write_log(logline)
    if which_spectra == 'total' or which_spectra == 'unlensed_total':
        deriv_name = "param_deriv/deriv_%s_all.npy"%(which_spectra)
        new_deriv_name = "param_deriv/new_deriv_dict_%s_all.npy"%(which_spectra)
    elif which_spectra == "delensed_scalar":
        deriv_name = "param_deriv/deriv_%s%s_n%s_all.npy"%(which_spectra, systype, rms_map_T_list[i])
        new_deriv_name = "param_deriv/new_deriv_dict_%s_%s%s_n%s_all.npy"%(itername,which_spectra, systype, rms_map_T_list[i])
    
    file_exists = exists(deriv_name)
    cl_deriv_dict = {}
    if file_exists:
        print('The deriv we already have')
        cl_deriv_dict = np.load(deriv_name)
        cl_deriv_dict = cl_deriv_dict.item()
    else:
        print('The deriv is new')
        cl_deriv_dict = tools.get_derivatives(param_dict, which_spectra, params_to_constrain = params_to_constrain,noise_nzero_fname = n0filename)
        np.save(deriv_name, cl_deriv_dict)

    print('finishi generate_derivatives','\n')


    file_exits = exists(new_deriv_name)
    new_deriv_dict = {}
    if file_exits:
        new_deriv_dict = np.load(new_deriv_name)
        new_deriv_dict = new_deriv_dict.item()
    else:
        new_deriv_dict = tools.rebin_deriv(els, cl_deriv_dict, binsize = binsize)
        np.save(new_deriv_name, new_deriv_dict)

    param_names = np.asarray( sorted( cl_deriv_dict.keys() ) )
    param_names.tolist()

    print('finishi rebin','\n')
    #continue
    ############################################################################################################

    ############################################################################################################

    Lsdl = 5
    binsize = 5   #rebin in calculating the cov^-1
    param_dict['binsize'] = binsize
    param_dict['Lsdl'] = Lsdl
    camborDl  = "Dl" # "camb" or "Dl"
    derivname = "selfdriv" # "camb" or "selfdriv"
    camborself = "self"
    print("dl is ", Lsdl)
    print("binsize is ", binsize)
    print("camborself is ", camborself)
    Ls_to_get = np.arange(2, 5000, Lsdl)

    
    ############################################################################################################
    #get delta_cl
    logline = '\tget delta Cl'; tools.write_log(logline)
    if which_spectra == 'total' or which_spectra == 'unlensed_total':
        delta_name = "diff_covariance/new_delta_dict_%s_n%s_addlensing.npy"%(which_spectra, rms_map_T_list[i])
    elif which_spectra == "delensed_scalar":
        delta_name = "diff_covariance/new_delta_dict_%s_%s%s_n%s_addlensing.npy"%(itername,which_spectra, sysornot, rms_map_T_list[i])
    file_exists = exists(delta_name)
    if file_exists:
        new_delta_dict = np.load(delta_name)
        new_delta_dict = new_delta_dict.item()
    else:
        new_delta_dict = tools.get_delta_cl_cov2(els, unlensedCL, cl_dict, nl_dict, which_spectra = which_spectra,dB_dE_dict = diff_EE_dict, diff_phi_dict = diff_phi_dict, diff_self_dict = diff_self_dict, Ls_to_get = Ls_to_get, min_l_temp = min_l_temp, max_l_temp = 3000)
        np.save(delta_name, new_delta_dict)

    gaussian_delta_name = "diff_covariance/new_gaussian_delta_dict_%s_n%s_addlensing.npy"%(which_spectra, rms_map_T_list[i])
    file_exists = exists(gaussian_delta_name)
    if file_exists:
        new_delta_dict_gaussian = np.load(gaussian_delta_name)
        new_delta_dict_gaussian = new_delta_dict_gaussian.item()
    else:
        new_delta_dict_gaussian = tools.get_gaussaian_cl_cov(els, cl_dict, nl_dict, min_l_temp = min_l_temp, max_l_temp = 3000)
        np.save(gaussian_delta_name, new_delta_dict_gaussian)

    print('finishi geting covariance','\n')

    ############################################################################################################
    #get Fisher

        
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    ############################################################################################################
    #get Fisher
    logline = '\tget fisher'; tools.write_log(logline)
    newl, F_mat, F_nongau_CMB, F_nongau_diag, F_nongau_ell_diag  = tools.get_fisher_mat_addlensing(els, new_deriv_dict, new_delta_dict, param_names, pspectra_to_use = pspectra_to_use,\
                                                                                                   min_l_temp = min_l_temp, max_l_temp = 3000, min_l_pol = min_l_pol, max_l_pol = 5000,binsize = binsize, addlensing = addlensing)


    np.save("ForNote/F_nongau_mat_%s_%s%s_n%s_%s.npy"%(itername,which_spectra, 'compall', rms_map_T_list[i], tail), F_mat)
    np.save("ForNote/F_nongau_CMB_%s_%s%s_n%s_%s.npy"%(itername,which_spectra, 'compall', rms_map_T_list[i], tail), F_nongau_CMB)

    newl, F_mat, F_CMB, F_diag, F_ell_diag  = tools.get_fisher_mat_addlensing(els, new_deriv_dict, new_delta_dict_gaussian, param_names, pspectra_to_use = pspectra_to_use,\
                                                                                                   min_l_temp = min_l_temp, max_l_temp = 3000, min_l_pol = min_l_pol, max_l_pol = 5000,binsize = binsize, addlensing = addlensing)


    np.save("ForNote/F_mat_%s_%s%s_n%s_%s.npy"%(itername,which_spectra, 'compall', rms_map_T_list[i], tail), F_mat)
    np.save("ForNote/F_CMB_%s_%s%s_n%s_%s.npy"%(itername,which_spectra, 'compall', rms_map_T_list[i], tail), F_CMB)


    F_mat = F_mat * fsky
    out_dict = {}
    out_dict['parms'] = param_names.tolist()
    out_dict['Fmat'] = F_mat.tolist()
    out_dict['fsky'] = fsky
    continue
