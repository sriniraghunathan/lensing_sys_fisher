"""
#CMB fisher forecasting
"""
############################################################################################################
#exec(open('plot_results.py').read()) 
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
import re

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
############################################################################################################
#get the necessary arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt')
#parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='delensed_scalar', choices=['lensed_scalar', 'unlensed_scalar', 'delensed_scalar', 'unlensed_total', 'total']) # add delensed
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

rms_map_T_list = np.array((0.1, 0.2, 0.5, 1, 2, 5, 10))#np.arange(1,11,1)
if rms_map_T == -1:
    rms_map_T_list = np.array((0.1, 0.2, 0.5, 1, 2, 5, 10))#np.arange(1,11,1)
else:
    rms_map_T_list = [rms_map_T]
fwhm_list = np.ones(len(rms_map_T_list))

############################################################################################################
#get experiment specs
logline = '\tset lmax / power spectra to be used'; tools.write_log(logline)

min_l_temp, max_l_temp = 30, 5000
min_l_pol, max_l_pol = 30, 5000
pspectra_to_use = ['TT', 'EE', 'TE', 'PP'] #CMB TT/EE/TE/lensing

############################################################################################################

############################################################################################################
params_to_constrain = ['As','tau','r','ns', 'ombh2', 'omch2', 'thetastar', 'gamma_phi_sys', 'gamma_N0_sys']
param_names = ['As', 'gamma_N0_sys', 'gamma_phi_sys', 'ns', 'ombh2', 'omch2', 'r', 'tau', 'thetastar']

fix_params = ['Alens', 'ws', 'omk']#, 'mnu'] #parameters to be fixed (if included in fisher forecasting)
#fix_params = ['r','ns', 'ombh2', 'thetastar']
#prior_dic = {'tau':0.007} #Planck-like tau prior
prior_dic = {}
if lensing_sys_n0_frac>0.:
    pass
    #prior_dic = {'Asys_lens':0.1, 'alphasys_lens': 0.1} #play with the choice of prior here.
desired_param_arr = None ##['ns', 'neff'] #desired parameter to be printed. Prints everything is set to None
############################################################################################################


############################################################################################################
#get/read the parameter file
logline = '\tget/read the parameter file'; tools.write_log(logline)
param_dict = tools.get_ini_param_dict(paramfile)
print('The parameters used are:\n')
print(param_dict)
param_dict['Alens'] = Alens
Lsdl = 5
binsize = 5   #rebin in calculating the cov^-1
param_dict['binsize'] = binsize
param_dict['Lsdl'] = Lsdl
els = np.arange(param_dict['min_l_limit'], param_dict['max_l_limit']+1)
systype = 'changefield'
systype = 'guessw'
iteration = True
itername = ''
if iteration:
    itername = 'iter1st'
sysornot = 'sys'
derivname = "selfdriv" # "camb" or "selfdriv"
camborself = "self"
print("dl is ", Lsdl)
print("binsize is ", binsize)
print("camborself is ", camborself)
Ls_to_get = np.arange(2, 5000, Lsdl)
            
#exit()
############################################################################################################

nl_dict = {}

clname = ['TT','EE','TE','BB']
names = ['TTTT','EEEE','TETE','BBBB']



for i in range(len(rms_map_T_list)):
    if i > 20:
        continue

    new_delta_dict = np.load("diff_covariance/new_delta_dict_%s_%s%s_n%s.npy"%(itername,which_spectra, sysornot, rms_map_T_list[i]))
    new_delta_dict = new_delta_dict.item()

    deriv_dict_filename = "param_deriv/new_deriv_dict_%s_%s%s_n%s.npy"%(itername,which_spectra, sysornot, rms_map_T_list[i])
    new_deriv_dict = np.load(deriv_dict_filename)
    new_deriv_dict = new_deriv_dict.item()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    ############################################################################################################
    #get Fisher
    logline = '\tget fisher'; tools.write_log(logline)
    newl, F_mat, F_nongau_CMB, F_nongau_ell, F_nongau_diag, F_nongau_ell_diag  = tools.get_fisher_mat_seperate(els, new_deriv_dict, new_delta_dict, param_names, pspectra_to_use = pspectra_to_use, min_l_temp = min_l_temp, max_l_temp = max_l_temp, min_l_pol = min_l_pol, max_l_pol = max_l_pol,binsize = binsize)


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    F_mat = F_mat * fsky
    out_dict = {}
    out_dict['parms'] = param_names
    out_dict['Fmat'] = F_mat.tolist()
    out_dict['fsky'] = fsky
    #print(out_dict.items())
    with open("results/F_mat_%s%s_bin%s_n%s_2gammas1.00_rem6_%s_cutBB30.json"%(itername, which_spectra, binsize, rms_map_T_list[i], systype), 'w') as fp:
            j = json.dump({k: v for k, v in out_dict.items()}, fp)


param_list = ['As', 'gamma_N0_sys', 'gamma_phi_sys', 'ns', 'ombh2', 'omch2', 'r', 'tau', 'thetastar']

