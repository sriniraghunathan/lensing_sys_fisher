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


############################################################################################################
#get the necessary arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt')
#parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='lensed_scalar', choices=['lensed_scalar', 'unlensed_scalar'])
parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='unlensed_scalar', choices=['lensed_scalar', 'unlensed_scalar', 'delensed_scalar']) # add delensed

#reduce lensing amplitude by xx per cent. Roughly mimicking S4-Wide delensing.
parser.add_argument('-Alens', dest='Alens', action='store', help='Alens', type=float, default=1) 
#parser.add_argument('-Alens', dest='Alens', action='store', help='Alens', type=float, default=0.3) 

parser.add_argument('-fsky', dest='fsky', action='store', help='fsky', type = float, default = 0.57)
parser.add_argument('-include_lensing', dest='include_lensing', action='store', help='include_lensing', type = int, default = 0)
#xx percent of lensing N0 will be considered as lensing systematic
parser.add_argument('-lensing_sys_n0_frac', dest='lensing_sys_n0_frac', action='store', help='lensing_sys_n0_frac', type = float, default = 0.2)

#ILC residual file
parser.add_argument('-use_ilc_nl', dest='use_ilc_nl', action='store', help='use_ilc_nl', type=int, default = 1)
#or add noise levels (only used if use_ilc_nl = 0)
parser.add_argument('-rms_map_T', dest='rms_map_T', action='store', help='rms_map_T', type=float, default = 2.)
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
params_to_constrain = ['As', 'neff', 'ns', 'ombh2', 'omch2', 'tau', 'thetastar', 'mnu']
#params_to_constrain = ['As','A_phi_sys', 'alpha_phi_sys', 'neff', 'ns', 'ombh2', 'omch2', 'tau', 'thetastar', 'mnu']
#params_to_constrain = ['As', 'neff', 'tau', 'thetastar', 'mnu']
#params_to_constrain = ['As', 'neff']
#params_to_constrain = ['neff']
#params_to_constrain = ['As','A_phi_sys', 'alpha_phi_sys', 'neff']
###params_to_constrain = ['As']
fix_params = ['Alens', 'ws', 'omk']#, 'mnu'] #parameters to be fixed (if included in fisher forecasting)
prior_dic = {'tau':0.007} #Planck-like tau prior
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
rms_map_P = rms_map_T * 1.414
Bl, nl_TT, nl_PP = tools.get_nl(els, rms_map_T, rms_map_P = rms_map_P, fwhm = fwhm_arcmins)
nl_dict['TT'] = nl_TT
nl_dict['EE'] = nl_PP
#nl_dict['TE'] = np.copy(nl_PP)
nl_dict['TE'] = np.copy(nl_PP)*0.
nl_dict['PP'] = np.zeros(len(nl_TT))

if use_ilc_nl:
    include_gal = 1
    gal_mask = 3 #only valid if galaxy is included
    if not include_gal:
        nlfile = '%s/S4_ilc_galaxy0_27-39-93-145-225-278_TT-EE-TE_lf2-mf12-hf5_AZ.npy' %(draft_results_folder)
    else:
        nlfile = '%s/S4_ilc_galaxy1_27-39-93-145-225-278_TT-EE-TE_lf2-mf12-hf5_galmask%s_AZ.npy' %(draft_results_folder, gal_mask)
    nl_dict = tools.get_nl_dict(nlfile, els)
    #nl_TT = nl_TT['TT']
    nl_TT = nl_dict['TT']
    nl_PP = nl_dict['EE']

'''
#add lensing systematic
if include_lensing and float(param_dict['A_phi_sys'])> 0.: #assume roughly xx per cent lensing N0 to be the systematic error

    els_pivot=3000
    #compute the lensing systematic in Cl space.
    factor_phi_deflection = (els * (els+1) )**2./2./np.pi
    A_phi_sys=float(param_dict['A_phi_sys'])
    alpha_phi_sys=float(param_dict['alpha_phi_sys'])
    #fit a power law for nl_mv_sys
    def get_nl_sys(A_phi_sys, alpha_phi_sys):
        nl_mv_sys = A_phi_sys *((els/ els_pivot)**alpha_phi_sys) * factor_phi_deflection
        return nl_mv_sys
    nl_mv_sys = get_nl_sys(A_phi_sys, alpha_phi_sys)
    print('nl_dict: ', nl_dict)
    if which_spectra == "delensed_scalar":
        nl_dict['PP'] += nl_mv_sys
        
        
    for ppp in ['A_phi_sys','alpha_phi_sys']:
        cl_deriv_dict[ppp] = {}
        #derivatives w.r.t TT/EE/TE/Tphi/EPhi are all zero since this is a lensing related systematic
        """
        Ahhhh - will Tphi actually be zero if this is some kind of foreground related systematic?
            CMB T x phi will be zero. But the total T will include foregrounds which could lead to non-zero Tphi(?)
            ignore that for now
        """

        cl_deriv_dict[ppp]['TT'] = np.zeros_like(els)
        cl_deriv_dict[ppp]['EE'] = np.zeros_like(els)
        cl_deriv_dict[ppp]['TE'] = np.zeros_like(els)
        cl_deriv_dict[ppp]['Tphi'] = np.zeros_like(els)
        cl_deriv_dict[ppp]['Ephi'] = np.zeros_like(els)
    
        if float(param_dict[ppp]) == 0.:
            step_size = 0.001 #some small number
        else:
            step_size = float(param_dict[ppp]) * 0.01 #1 per cent for the original parameter.
        ppp_low_val, ppp_high_val = float(param_dict[ppp]) - step_size, float(param_dict[ppp]) + step_size
        if ppp == 'A_phi_sys':
            nl_mv_sys_low = get_nl_sys(ppp_low_val, alpha_phi_sys)
            nl_mv_sys_high = get_nl_sys(ppp_high_val, alpha_phi_sys)
        elif ppp == 'alpha_phi_sys':
            nl_mv_sys_low = get_nl_sys(A_phi_sys, ppp_low_val)
            nl_mv_sys_high = get_nl_sys(A_phi_sys, ppp_high_val)
        nl_mv_sys_der = (nl_mv_sys_high - nl_mv_sys_low) / ( 2 * step_size)
        cl_deriv_dict[ppp]['PP']  = nl_mv_sys_der # don't think this is needed * factor_phi_deflection 


    #modify param_names to include lensing related systematic
    #print(param_names)
    param_names = np.asarray( sorted( cl_deriv_dict.keys() ) )
    print('The considered parameters are', param_names,'\n')
    print("The noise level is %f"%(rms_map_T))
'''
if which_spectra == "delensed_scalar":
    nels = np.arange(els[0], els[-1]+10, 100)
    #La = np.arange(els[0], 50, 1)
    #Lb = np.arange(50, 300, 10)
    #Lc = np.arange(300, els[-1]+10, 100)
    #L1 = np.append(La, Lb)
    #L = np.append(L1, Lc)

    n0s = tools.calculate_n0(nels, els, unlensedCL, totCL, nl_TT, nl_PP, dimx = 1024, dimy = 1024, fftd = 1./60/180)
    mv = 1./(1./n0s['EB']+1./n0s['EE']+1./n0s['TT']+1./n0s['TB']+1./n0s['TE'])
    data = np.column_stack((nels,n0s['EB'],n0s['EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv))
    header = "els,n0s['EB'],n0s[q'EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv" 
    output_name = "params/generate_n0s_rmsT%s_fwhmm%s.dat"%(rms_map_T, fwhm_arcmins)
    np.savetxt(output_name, data, header=header)
    param_dict['rms_map_T'] = rms_map_T
    param_dict['rms_map_P'] = rms_map_P
    param_dict['nlP'] = nl_PP
    param_dict['nlT'] = nl_TT
    param_dict['fwhm_arcmins'] = fwhm_arcmins
    n0_els, n0_mv = nels, mv
    nl_mv = interpolate.interp1d(nels, mv)
    #nl_mv = np.interp(els, n0_els, n0_mv, left = 1e6, right = 1e6) #set noise of els beyond n0_els to some large number
    nl_dict['PP'] = nl_mv(els)




print('nl_dict: ', nl_dict)
'''
if 'PP' in pspectra_to_use: #include lensing noise
    lensing_n0_fname = 'lensing_noise_curves/S4_ilc_galaxy0_27-39-93-145-225-278_TT-EE-TE_lf2-mf12-hf5_AZ_lmin100_lmax5000_lmaxtt3000.npy'
    lensing_n0_dict = np.load(lensing_n0_fname, allow_pickle=True, encoding='latin1').item()
    n0_els, n0_mv = lensing_n0_dict['els'], lensing_n0_dict['Nl_MV'].real
    n0_mv[np.isnan(n0_mv)] = 0.
    nl_mv = np.interp(els, n0_els, n0_mv, left = 1e6, right = 1e6) #set noise of els beyond n0_els to some large number
    nl_dict['PP'] = nl_mv
'''
#please change to get the delensed spectra    
############################################################################################################

############################################################################################################
#get fiducual LCDM power spectra computed using CAMB

logline = '\tget fiducual LCDM %s power spectra computed using CAMB' %(which_spectra); tools.write_log(logline)
pars, els, cl_dict = tools.get_cmb_spectra_using_camb(param_dict, which_spectra)

if (1):#debug:
    ax = plt.subplot(111, yscale = 'log')
    dl_fac = els * (els+1)/2/np.pi
    plt.plot(els, dl_fac * cl_dict['TT'], 'black'); 
    plt.plot(els, dl_fac * cl_dict['EE'], 'darkred');
    plt.plot(els, dl_fac * cl_dict['TE'], 'orangered');
    plt.plot(els, dl_fac * cl_dict['BB'], 'blue');
    plt.ylim(1e-3, 1e4)
    #xlabel(r'Multipole $\ell$'); ylabel(r'D$_{\ell}$ [$\mu$K$^{2}$]')
    #show()
    print("here")
    plt.savefig("power_camb2.png")
    #sys.exit()
############################################################################################################

############################################################################################################
#get derivatives
#please change to get the derivatives for the delensed spectra
logline = '\tget/read derivatives'; tools.write_log(logline)
cl_deriv_dict = tools.get_derivatives(param_dict, which_spectra, params_to_constrain = params_to_constrain)
param_names = np.asarray( sorted( cl_deriv_dict.keys() ) )
if (0):#debug:
    for keyname in cl_deriv_dict:
        ax = subplot(111, yscale = 'log')
        els = np.arange(len(cl_deriv_dict[keyname]['TT']))
        dl_fac = els * (els+1)/2/np.pi
        print(cl_deriv_dict[keyname]['TT'])
        plot(els, dl_fac * cl_deriv_dict[keyname]['TT'], 'black'); 
        plot(els, dl_fac * cl_deriv_dict[keyname]['EE'], 'darkred');
        plot(els, dl_fac * cl_deriv_dict[keyname]['TE'], 'orangered');

        title(keyname)
        xlabel(r'Multipole $\ell$'); ylabel(r'd$C_{\ell}^{XX}$/d$%s$' %(keyname.replace('_','\_')))
        show()
    #sys.exit()
############################################################################################################

print('nl_dict: ', nl_dict)
print(' cl_deriv_dict :', cl_deriv_dict)

############################################################################################################
############################################################################################################

############################################################################################################
#get delta_cl
logline = '\tget delta Cl'; tools.write_log(logline)
delta_cl_dict = tools.get_delta_cl_cov(els, cl_dict, nl_dict, include_lensing = include_lensing)
#delta_cl_dict = tools.get_delta_cl(els, cl_dict, nl_dict, include_lensing = include_lensing)
############################################################################################################

############################################################################################################
#get Fisher
logline = '\tget fisher'; tools.write_log(logline)
F_mat = tools.get_fisher_mat2(els, cl_deriv_dict, delta_cl_dict, param_names, pspectra_to_use = pspectra_to_use,\
            min_l_temp = min_l_temp, max_l_temp = max_l_temp, min_l_pol = min_l_pol, max_l_pol = max_l_pol)
#print(F_mat); sys.exit()
############################################################################################################
print(' F_mat: ', F_mat)

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
#print(param_names); sys.exit()
############################################################################################################

############################################################################################################
#add prior
logline = '\tadding prior'; tools.write_log(logline)
F_mat = tools.add_prior(F_mat, param_names, prior_dic)
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
with open('results_%s_%s.txt'%(which_spectra, rms_map_T),'w') as outfile:
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

#sys.exit()


ax = plt.subplot(111, yscale = 'log')
dl_fac = els * (els+1)/2/np.pi
dneff = cl_deriv_dict['neff']

plt.figure(figsize = (8,5))
plt.plot(els, dl_fac * cl_dict['TT'], 'black'); 
plt.fill_between(els, dl_fac*(cl_dict['TT'] - delta_cl_dict['TT']), dl_fac*(cl_dict['TT'] + delta_cl_dict['TT']), color = 'black', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['EE'], 'green'); 
plt.fill_between(els, dl_fac*(cl_dict['EE'] - delta_cl_dict['EE']), dl_fac*(cl_dict['EE'] + delta_cl_dict['EE']), color = 'green', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['TE'], 'orangered'); 
plt.fill_between(els, dl_fac*(cl_dict['TE'] - delta_cl_dict['TE']), dl_fac*(cl_dict['TE'] + delta_cl_dict['TE']), color = 'orangered', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['BB'], 'blue'); 
plt.fill_between(els, dl_fac*(cl_dict['BB'] - delta_cl_dict['BB']), dl_fac*(cl_dict['BB'] + delta_cl_dict['BB']), color = 'blue', alpha=0.5)

plt.plot(els, dl_fac*dneff['TT'], linestyle = '--', color = 'black')
plt.plot(els, dl_fac*dneff['EE'], linestyle = '--', color = 'green')
plt.plot(els, dl_fac*dneff['TE'], linestyle = '--', color = 'orangered')
plt.plot(els, dl_fac*dneff['BB'], linestyle = '--', color = 'blue')
#plt.ylim(1e-3, 1e4)
plt.xscale('log')
plt.yscale('log')

print("here")
plt.savefig("dneff_error.png")


plt.figure(figsize = (8,5))
plt.plot(els, dl_fac * cl_dict['TT'], 'black'); 
plt.fill_between(els, dl_fac*(cl_dict['TT'] - delta_cl_dict['TT']), dl_fac*(cl_dict['TT'] + delta_cl_dict['TT']), color = 'black', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['EE'], 'green'); 
plt.fill_between(els, dl_fac*(cl_dict['EE'] - delta_cl_dict['EE']), dl_fac*(cl_dict['EE'] + delta_cl_dict['EE']), color = 'green', alpha=0.5)
plt.plot(els, abs(dl_fac * cl_dict['TE']), 'orangered'); 
plt.fill_between(els, abs(dl_fac*(cl_dict['TE'] - delta_cl_dict['TE'])),abs( dl_fac*(cl_dict['TE'] + delta_cl_dict['TE'])), color = 'orangered', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['BB'], 'blue'); 
plt.fill_between(els, dl_fac*(cl_dict['BB'] - delta_cl_dict['BB']), dl_fac*(cl_dict['BB'] + delta_cl_dict['BB']), color = 'blue', alpha=0.5)

plt.plot(els, abs(dl_fac*dneff['TT']), linestyle = '--', color = 'black')
plt.plot(els, abs(dl_fac*dneff['EE']), linestyle = '--', color = 'green')
plt.plot(els, abs(dl_fac*dneff['TE']), linestyle = '--', color = 'orangered')
plt.plot(els, abs(dl_fac*dneff['BB']), linestyle = '--', color = 'blue')
#plt.ylim(1e-3, 1e4)
plt.xscale('log')
plt.yscale('log')

print("here")
plt.savefig("dneff_error3.png")

dAs = cl_deriv_dict['As']
plt.figure(figsize = (8,5))
plt.plot(els, dl_fac * cl_dict['TT'], 'black'); 
plt.fill_between(els, dl_fac*(cl_dict['TT'] - delta_cl_dict['TT']), dl_fac*(cl_dict['TT'] + delta_cl_dict['TT']), color = 'black', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['EE'], 'green'); 
plt.fill_between(els, dl_fac*(cl_dict['EE'] - delta_cl_dict['EE']), dl_fac*(cl_dict['EE'] + delta_cl_dict['EE']), color = 'green', alpha=0.5)
plt.plot(els, abs(dl_fac * cl_dict['TE']), 'orangered'); 
plt.fill_between(els, abs(dl_fac*(cl_dict['TE'] - delta_cl_dict['TE'])),abs( dl_fac*(cl_dict['TE'] + delta_cl_dict['TE'])), color = 'orangered', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['BB'], 'blue'); 
plt.fill_between(els, dl_fac*(cl_dict['BB'] - delta_cl_dict['BB']), dl_fac*(cl_dict['BB'] + delta_cl_dict['BB']), color = 'blue', alpha=0.5)

plt.plot(els, abs(dl_fac*dAs['TT']), linestyle = '--', color = 'black')
plt.plot(els, abs(dl_fac*dAs['EE']), linestyle = '--', color = 'green')
plt.plot(els, abs(dl_fac*dAs['TE']), linestyle = '--', color = 'orangered')
plt.plot(els, abs(dl_fac*dAs['BB']), linestyle = '--', color = 'blue')
#plt.ylim(1e-3, 1e4)
plt.xscale('log')
plt.yscale('log')

print("here")
plt.savefig("dAs_error3.png")

#params_to_constrain = ['As', 'neff', 'tau', 'thetastar', 'mnu']


dtau = cl_deriv_dict['tau']
plt.figure(figsize = (8,5))
plt.plot(els, dl_fac * cl_dict['TT'], 'black'); 
plt.fill_between(els, dl_fac*(cl_dict['TT'] - delta_cl_dict['TT']), dl_fac*(cl_dict['TT'] + delta_cl_dict['TT']), color = 'black', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['EE'], 'green'); 
plt.fill_between(els, dl_fac*(cl_dict['EE'] - delta_cl_dict['EE']), dl_fac*(cl_dict['EE'] + delta_cl_dict['EE']), color = 'green', alpha=0.5)
plt.plot(els, abs(dl_fac * cl_dict['TE']), 'orangered'); 
plt.fill_between(els, abs(dl_fac*(cl_dict['TE'] - delta_cl_dict['TE'])),abs( dl_fac*(cl_dict['TE'] + delta_cl_dict['TE'])), color = 'orangered', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['BB'], 'blue'); 
plt.fill_between(els, dl_fac*(cl_dict['BB'] - delta_cl_dict['BB']), dl_fac*(cl_dict['BB'] + delta_cl_dict['BB']), color = 'blue', alpha=0.5)

plt.plot(els, abs(dl_fac*dtau['TT']), linestyle = '--', color = 'black')
plt.plot(els, abs(dl_fac*dtau['EE']), linestyle = '--', color = 'green')
plt.plot(els, abs(dl_fac*dtau['TE']), linestyle = '--', color = 'orangered')
plt.plot(els, abs(dl_fac*dtau['BB']), linestyle = '--', color = 'blue')
#plt.ylim(1e-3, 1e4)
plt.xscale('log')
plt.yscale('log')

print("here")
plt.savefig("dtau_error3.png")


dmnu = cl_deriv_dict['mnu']
plt.figure(figsize = (8,5))
plt.plot(els, dl_fac * cl_dict['TT'], 'black'); 
plt.fill_between(els, dl_fac*(cl_dict['TT'] - delta_cl_dict['TT']), dl_fac*(cl_dict['TT'] + delta_cl_dict['TT']), color = 'black', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['EE'], 'green'); 
plt.fill_between(els, dl_fac*(cl_dict['EE'] - delta_cl_dict['EE']), dl_fac*(cl_dict['EE'] + delta_cl_dict['EE']), color = 'green', alpha=0.5)
plt.plot(els, abs(dl_fac * cl_dict['TE']), 'orangered'); 
plt.fill_between(els, abs(dl_fac*(cl_dict['TE'] - delta_cl_dict['TE'])),abs( dl_fac*(cl_dict['TE'] + delta_cl_dict['TE'])), color = 'orangered', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['BB'], 'blue'); 
plt.fill_between(els, dl_fac*(cl_dict['BB'] - delta_cl_dict['BB']), dl_fac*(cl_dict['BB'] + delta_cl_dict['BB']), color = 'blue', alpha=0.5)

plt.plot(els, abs(dl_fac*dmnu['TT']), linestyle = '--', color = 'black')
plt.plot(els, abs(dl_fac*dmnu['EE']), linestyle = '--', color = 'green')
plt.plot(els, abs(dl_fac*dmnu['TE']), linestyle = '--', color = 'orangered')
plt.plot(els, abs(dl_fac*dmnu['BB']), linestyle = '--', color = 'blue')
#plt.ylim(1e-3, 1e4)
plt.xscale('log')
plt.yscale('log')

print("here")
plt.savefig("dmnu_error3.png")


dthetastar = cl_deriv_dict['thetastar']
plt.figure(figsize = (8,5))
plt.plot(els, dl_fac * cl_dict['TT'], 'black'); 
plt.fill_between(els, dl_fac*(cl_dict['TT'] - delta_cl_dict['TT']), dl_fac*(cl_dict['TT'] + delta_cl_dict['TT']), color = 'black', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['EE'], 'green'); 
plt.fill_between(els, dl_fac*(cl_dict['EE'] - delta_cl_dict['EE']), dl_fac*(cl_dict['EE'] + delta_cl_dict['EE']), color = 'green', alpha=0.5)
plt.plot(els, abs(dl_fac * cl_dict['TE']), 'orangered'); 
plt.fill_between(els, abs(dl_fac*(cl_dict['TE'] - delta_cl_dict['TE'])),abs( dl_fac*(cl_dict['TE'] + delta_cl_dict['TE'])), color = 'orangered', alpha=0.5)
plt.plot(els, dl_fac * cl_dict['BB'], 'blue'); 
plt.fill_between(els, dl_fac*(cl_dict['BB'] - delta_cl_dict['BB']), dl_fac*(cl_dict['BB'] + delta_cl_dict['BB']), color = 'blue', alpha=0.5)

plt.plot(els, abs(dl_fac*dthetastar['TT']), linestyle = '--', color = 'black')
plt.plot(els, abs(dl_fac*dthetastar['EE']), linestyle = '--', color = 'green')
plt.plot(els, abs(dl_fac*dthetastar['TE']), linestyle = '--', color = 'orangered')
plt.plot(els, abs(dl_fac*dthetastar['BB']), linestyle = '--', color = 'blue')
#plt.ylim(1e-3, 1e4)
plt.xscale('log')
plt.yscale('log')

print("here")
plt.savefig("dthetastar_error3.png")

plt.figure(figsize = (8,5))
plt.plot(els, abs(dl_fac*dneff['TT']), linestyle = '--', color = 'black')
plt.plot(els, abs(dl_fac*dneff['EE']), linestyle = '--', color = 'green')
plt.plot(els, abs(dl_fac*dneff['TE']), linestyle = '--', color = 'orangered')
plt.plot(els, abs(dl_fac*dneff['BB']), linestyle = '--', color = 'blue')

plt.plot(els, abs(dl_fac*delta_cl_dict['TT']), color = 'black')
plt.plot(els, abs(dl_fac*delta_cl_dict['EE']), color = 'green')
plt.plot(els, abs(dl_fac*delta_cl_dict['TE']), color = 'orangered')
plt.plot(els, abs(dl_fac*delta_cl_dict['BB']), color = 'blue')
print("here")
plt.xscale('log')
plt.yscale('log')
plt.savefig("dneff_error2.png")


plt.figure(figsize = (8,5))
plt.plot(els, abs(dneff['TT']/delta_cl_dict['TT']), linestyle = '--', color = 'black')
plt.plot(els, abs(dneff['EE']/delta_cl_dict['TT']), linestyle = '--', color = 'green')
plt.plot(els, abs(dneff['TE']/delta_cl_dict['TT']), linestyle = '--', color = 'orangered')
plt.plot(els, abs(dneff['BB']/delta_cl_dict['TT']), linestyle = '--', color = 'blue')

print("here")
plt.xscale('log')
plt.yscale('log')
plt.savefig("dneff_error4.png")





