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

############################################################################################################
#get the necessary arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt')
#parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='delensed_scalar', choices=['lensed_scalar', 'unlensed_scalar', 'delensed_scalar', 'unlensed_total', 'total']) # add delensed
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
#parser.add_argument('-rms_map_T_list', dest='rms_map_T_list', action='store',help='rms_map_T_list',type=float, default = np.arange(1,11,1))
#parser.add_argument('-rms_map_T_list', dest='rms_map_T_list', action='store',help='rms_map_T_list',type=<class 'numpy.ndarray'>, default = np.arange(1,11,1))
#parser.add_argument('-fwhm_list', dest='fwhm_list', action='store',help='fwhm_list',type=<class 'numpy.ndarray'>, default = np.ones(10))

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
    rms_map_T_list = np.arange(1,11,1)
else:
    rms_map_T_list = [rms_map_T]
fwhm_list = np.ones(10)


Lsdl = 5
binsize = 5   #rebin in calculating the cov^-1

if not os.path.exists('derivs/'): os.system('mkdir -p derivs')
############################################################################################################
#get experiment specs
logline = '\tset lmax / power spectra to be used'; tools.write_log(logline)
if include_lensing:
    pspectra_to_use = ['TT', 'EE', 'TE', 'PP'] #CMB TT/EE/TE/lensing
else:
    pspectra_to_use = ['TT', 'EE', 'TE'] #CMB TT/EE/TE
min_l_temp, max_l_temp = 30, 5000
min_l_pol, max_l_pol = 30, 5000
############################################################################################################

############################################################################################################
#cosmological parameters
#params_to_constrain = ['As', 'ns', 'ombh2', 'omch2', 'tau', 'mnu', 'r']
#params_to_constrain = ['As', 'neff', 'ns', 'ombh2', 'omch2', 'tau', 'thetastar', 'mnu']
#params_to_constrain = ['neff','ns', 'ombh2', 'omch2', 'thetastar', 'A_phi_sys', 'alpha_phi_sys']
###params_to_constrain = ['r','ns', 'ombh2', 'omch2', 'thetastar', 'A_phi_sys', 'alpha_phi_sys']
#params_to_constrain = ['As','tau','r','ns', 'ombh2', 'omch2', 'thetastar','A_phi_sys', 'alpha_phi_sys']
params_to_constrain = ['As','tau','r','ns', 'ombh2', 'omch2', 'thetastar']
#params_to_constrain = ['r','ns', 'ombh2', 'omch2', 'thetastar']
#params_to_constrain = ['r', 'thetastar']
#params_to_constrain = ['neff','ns', 'ombh2', 'omch2', 'thetastar']
#params_to_constrain = ['neff', 'thetastar']
fix_params = ['Alens', 'ws', 'omk']#, 'mnu'] #parameters to be fixed (if included in fisher forecasting)
#fix_params = ['r','ns', 'ombh2', 'thetastar']
#prior_dic = {'tau':0.007} #Planck-like tau prior
#prior_dic = {'tau':0.002} #Planck-like tau prior
prior_dic = {}
#prior_dic = {'A_phi_sys':5e-19, 'alpha_phi_sys':0.2} #Planck-like tau prior
#prior_dic = {'tau':0.002, 'A_phi_sys':5e-19, 'alpha_phi_sys':0.2} #Planck-like tau prior
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

for i in range(len(rms_map_T_list)):
    opfname = 'results/F_mat_CDMp_test_%s_bin%s_dl%d_%s_n%.1f.json' %(which_spectra, binsize, Lsdl, camborself, rms_map_T_list[i])
    if os.path.exists(opfname):
        print('\n\tAlready complete. Check %s.\n' %(opfname))
        continue

    noise_nzero_fname = 'params/generate_n0s_rmsT%.1f_fwhmm%.1f_dl%d.dat'%(rms_map_T_list[i], 1.0, binsize)
    file_exists = exists(noise_nzero_fname)
    print('Calculating N0 for noise%s'%(rms_map_T_list[i]))
    #if which_spectra == "delensed_scalar":
    if file_exists:
        print('\tAlready have N0 for this noise level. File is %s!!! \n' %(noise_nzero_fname))
        n0s = np.loadtxt(noise_nzero_fname)
        mv = n0s[:,-1]
        nels = n0s[:,0]
        nl_mv = interpolate.interp1d(nels, mv)
        nl_dict['PP'] = nl_mv(els)
    else:
        nels = np.arange(els[0], els[-1]+10, 10)
        n0s = tools.calculate_n0(nels, els, unlensedCL, totCL, nl_TT_list[i], nl_PP_list[i], dimx = 1024, dimy = 1024, fftd = 1./60/180)
        mv = 1./(1./n0s['EB']+1./n0s['EE']+1./n0s['TT']+1./n0s['TB']+1./n0s['TE'])
        data = np.column_stack((nels,n0s['EB'],n0s['EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv))
        header = "els,n0s['EB'],n0s[q'EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv" 
        np.savetxt(noise_nzero_fname, data, header=header)
        n0_els, n0_mv = nels, mv
        nl_mv = interpolate.interp1d(nels, mv)
        nl_dict['PP'] = nl_mv(els)


    param_dict['rms_map_T'] = rms_map_T_list[i]
    param_dict['rms_map_P'] = rms_map_T_list[i] * 2**0.5
    param_dict['nlP'] = nl_PP_list[i]
    param_dict['nlT'] = nl_TT_list[i]
    param_dict['fwhm_arcmins'] = fwhm_list[i]
    param_dict['binsize'] = binsize

    nl_dict['TT'] = nl_TT_list[i]
    nl_dict['EE'] = nl_PP_list[i]
    nl_dict['BB'] = nl_PP_list[i]
    nl_dict['TE'] = nl_TT_list[i]*0
    nl_dict['SYS'] = tools.get_nl_sys(els, param_dict['A_phi_sys'], param_dict['alpha_phi_sys'])
    

    ############################################################################################################
    #get fiducual LCDM power spectra computed using CAMB
    logline = '\tget fiducual LCDM %s power spectra computed using CAMB' %(which_spectra); tools.write_log(logline)
    pars, els, cl_dict = tools.get_cmb_spectra_using_camb(param_dict, which_spectra, noise_nzero_fname = noise_nzero_fname)
    ############################################################################################################

    ############################################################################################################
    #get derivatives
    logline = '\tget/read derivatives'; tools.write_log(logline)
    cl_deriv_dict = tools.get_derivatives(param_dict, which_spectra, params_to_constrain = params_to_constrain, noise_nzero_fname = noise_nzero_fname)
    param_names = np.asarray( sorted( cl_deriv_dict.keys() ) )
    ############################################################################################################

    ############################################################################################################
    #get delta_cl
    logline = '\tget delta Cl'; tools.write_log(logline)
    #delta_cl_dict = tools.get_delta_cl_cov(els, cl_dict, nl_dict, include_lensing = include_lensing, which_spectra = which_spectra, include_B = False)
    print('cl_dict.keys ', cl_dict.keys(), '\n')
    print('nl_dict.keys ', nl_dict.keys(), '\n')
    ############################################################################################################

    '''
    add non_gaussian term, with derivatives to Cl^phiphi
    els, powerini = tools.get_ini_cmb_power(param_dict, raw_cl = 1)
    unlensedCL = powerini['unlensed_total']
    totCL = powerini['total']
    cl_phiphi, cl_Tphi, cl_Ephi = powerini['lens_potential'].T
    cphifun = interpolate.interp1d(els, cl_phiphi)
    '''
    

    #'''
    cov_nongaussian = {}
    camborDl  = "Dl" # "camb" or "Dl"
    #derivname = "selfdriv" # "camb" or "selfdriv"
    #camborself = "self"
    derivname = "selfdriv" # "camb" or "selfdriv"
    camborself = "self"
    print("dl is ", Lsdl)
    print("binsize is ", binsize)
    print("camborself is ", camborself)
    Ls_to_get = np.arange(2, 5000, Lsdl)
    file_exists = exists("derivs/diffphi_dl%s_%s.json"%(Lsdl, camborDl))
    #file_exists = False
    if which_spectra == "delensed_scalar":        
        file_exists = exists("derivs/diffphi_dl%s_Dl_delensed_n%s.json"%(Lsdl, rms_map_T_list[i]))
        #file_exists = False

    diff_EE_dict = {}; diff_self_dict = {}; diff_phi_dict = {}

    if file_exists:
        if which_spectra == "delensed_scalar":        
            print("Already have derivs for this dl!!! \n" , "dl = %s"%(Lsdl))
            with open("derivs/diffphi_dl%s_Dl_delensed_n%s.json"%(Lsdl, rms_map_T_list[i])) as infile:
                diff_phi_data = json.load(infile)
            with open("derivs/diffself_dl%s_Dl_delensed_n%s.json"%(Lsdl, rms_map_T_list[i])) as infile:
                diff_self_data = json.load(infile)
            with open("derivs/diffee_dl%s_Dl_delensed_n%s.json"%(Lsdl, rms_map_T_list[i])) as infile:
                diff_EE_data = json.load(infile)
        else:
            print("Already have derivs for this dl!!! \n" , "dl = %s"%(Lsdl))
            with open("derivs/diffphi_dl%s_%s.json"%(Lsdl, camborDl)) as infile:
                diff_phi_data = json.load(infile)
            with open("derivs/diffself_dl%s_%s.json"%(Lsdl, camborDl)) as infile:
                diff_self_data = json.load(infile)
            with open("derivs/diffee_dl%s_%s.json"%(Lsdl, camborDl)) as infile:
                diff_EE_data = json.load(infile)

        diff_EE_dict['BB'] = np.asarray(diff_EE_data['BB'])
        diff_self_dict['TT'] = np.asarray(diff_self_data['TT'])
        diff_self_dict['EE'] = np.asarray(diff_self_data['EE'])
        diff_self_dict['TE'] = np.asarray(diff_self_data['TE'])
        diff_phi_dict['TT'] = np.asarray(diff_phi_data['TT'])
        diff_phi_dict['EE'] = np.asarray(diff_phi_data['EE'])
        diff_phi_dict['BB'] = np.asarray(diff_phi_data['BB'])
        diff_phi_dict['TE'] = np.asarray(diff_phi_data['TE'])

    else:
        print("Calculate diffphi, diffe, diffself!")
        if camborDl == "camb":
            diff_EE_dict, diff_phi_dict, diff_self_dict = tools.get_deriv_camb(which_spectra, els, unlensedCL, cl_phiphi, nl_dict, Ls_to_get = Ls_to_get, percent=0.05)
        else:
            diff_EE_dict, diff_phi_dict, diff_self_dict = tools.get_deriv_clBB(which_spectra, els, unlensedCL, cl_phiphi, nl_dict, Ls_to_get = Ls_to_get, percent=0.05, noiseTi = rms_map_T_list[i])
    
    
    delta_cl_dict = tools.get_delta_cl_cov(els, cl_dict, nl_dict, fsky = 1., include_lensing = False, include_B = True, dB_dE_dict = diff_EE_dict, diff_phi_dict = diff_phi_dict, diff_self_dict = diff_self_dict, which_spectra = which_spectra, Ls_to_get = Ls_to_get)
    #diff_dict = tools.get_derivative_to_phi_with_camb(els, which_spectra, unlensedCL, cl_phiphi, nl_dict, Ls_to_get, percent=0.05)
    #delta_cl_dict_nongau = tools.get_nongaussaian_cl_cov(which_spectra, Ls_to_get, diff_dict, delta_cl_dict, els, cl_phiphi, nl_dict)

    #'''

    ############################################################################################################
    #get Fisher
    logline = '\tget fisher'; tools.write_log(logline)
    F_mat, covmat, [new_delta_dict, new_deriv_dict] = tools.get_fisher_mat5(els, cl_deriv_dict, delta_cl_dict, param_names, pspectra_to_use = pspectra_to_use,\
                                            min_l_temp = min_l_temp, max_l_temp = max_l_temp, min_l_pol = min_l_pol, max_l_pol = max_l_pol, delta_cl_dict_nongau = None, binsize = binsize, include_B = True)
    '''
    Fttmat, Feemat, Ftteemat = tools.get_fisher_mat3(els, cl_deriv_dict, delta_cl_dict, param_names, pspectra_to_use = pspectra_to_use,\
                                            min_l_temp = min_l_temp, max_l_temp = max_l_temp, min_l_pol = min_l_pol, max_l_pol = max_l_pol, delta_cl_dict_nongau = None)
    '''

    ############################################################################################################
    #add fsky to Fisher
    logline = '\tadd fsky to Fisher'; tools.write_log(logline)
    F_mat = F_mat * fsky
    ############################################################################################################

    ############################################################################################################
    #fix params
    logline = '\tfixing paramaters, if need be'; tools.write_log(logline)
    F_mat, param_names = tools.fix_params(F_mat, param_names, fix_params)
    param_names = np.asarray(param_names)
    ############################################################################################################

    ############################################################################################################
    #add prior
    logline = '\tadding prior'; tools.write_log(logline)
    F_mat = tools.add_prior(F_mat, param_names, prior_dic)
    ############################################################################################################

    ############################################################################################################
    if (1): #20220915 - debug: fix all but one parameter and check constraints.
        logline = '\tdebugging Fisher matrix for individual parameter constraints'; tools.write_log(logline)
        F_mat_for_debug = np.copy(F_mat)
        param_names_for_debug = np.copy(param_names)
        for curr_param in param_names_for_debug:
            params_to_fix = np.setxor1d( param_names_for_debug, [curr_param])
            curr_param_F_mat, curr_param_names = tools.fix_params(F_mat_for_debug, param_names_for_debug, params_to_fix)
            curr_cov_mat = 1./curr_param_F_mat
            sigma = curr_cov_mat**0.5
            opline = '\t\tsigma(%s) = %g using %s; fsky = %s; power spectra = %s (Alens = %s)' %(curr_param, sigma, str(pspectra_to_use), fsky, which_spectra, Alens)
            print(opline)
    ############################################################################################################


    ############################################################################################################
    #get cov matrix now
    logline = '\tget covariance matrix'; tools.write_log(logline)
    #cov_mat = sc.linalg.pinv2(F_mat) #made sure that COV_mat_l * Cinv_l ~= I
    cov_mat = np.linalg.inv(F_mat) #made sure that COV_mat_l * Cinv_l ~= I
    ############################################################################################################

    ############################################################################################################
    #extract parameter constraints
    if desired_param_arr is None:
        desired_param_arr = param_names
    opfname_constraints = 'results/constraints_CDMp_test_inv_bin%s_BB_%s_cut30_%s_dl%d_n%.1f_fwhm%s.txt'%(binsize, derivname,  which_spectra, Lsdl, rms_map_T_list[i], fwhm_list[i])
    with open(opfname_constraints,'w') as outfile:
        outfile.write('sigma,value\n')
        for desired_param in desired_param_arr:
            logline = '\textract sigma(%s)' %(desired_param); tools.write_log(logline)
            pind = np.where(param_names == desired_param)[0][0]
            pcntr1, pcntr2 = pind, pind
            print('pind ',pind, 'pcntr1',pcntr1,'pcntr2',pcntr2)
            cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]
            cov_extract = np.asarray( [cov_mat[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
            print('cov_extract: ',cov_extract)
            sigma = cov_extract[0,0]**0.5
            opline = '\t\t\sigma(%s) = %g using %s; fsky = %s; power spectra = %s (Alens = %s)' %(desired_param, sigma, str(pspectra_to_use), fsky, which_spectra, Alens)
            print(opline)
            outfile.write('sigma(%s),%g\n'%(desired_param, sigma))
    ############################################################################################################


    out_dict = {}
    out_dict['parms'] = param_names.tolist()
    out_dict['Fmat'] = F_mat.tolist()
    out_dict['cov_mat'] = cov_mat.tolist()
    out_dict['fsky'] = fsky

    with open(opfname, 'w') as fp:
            j = json.dump({k: v for k, v in out_dict.items()}, fp)

print('\nAll done.\n')
#sys.exit()
#ax = plt.subplot(111, yscale = 'log')
#dl_fac = els * (els+1)/2/np.pi
#dneff = cl_deriv_dict['neff']

