#################################################
#make corner plot using tools_for_plotting.py
import numpy as np, sys, os, glob, json
import tools_for_plotting, tools
from pylab import *

#Fisher matrix file names
reqd_fsky = None #0.03
tau_prior = 0.007 #None #0.007 #None #0.007
#tau_prior = None #0.007
#noise_level = 6
noise_level = 1
'''
flist_dict = {'results/F_mat_CDMp_total_bin5_dl5_self_n%s.json' %(noise_level): 'lensed', 
            'results/F_mat_CDMp_delensed_scalar_bin5_dl5_self_n%s.json' %(noise_level): 'delensed', 
            'results/F_mat_CDMp_unlensed_total_bin5_dl5_self_n%s.json' %(noise_level): 'unlensed'}
'''
'''
flist_dict = {'results/F_mat_CDMp_test_total_bin5_dl5_self_n%s.json' %(noise_level): 'lensed', 
            'results/F_mat_CDMp_test_delensed_scalar_bin5_dl5_self_n%s.json' %(noise_level): 'delensed', 
            'results/F_mat_CDMp_test_unlensed_total_bin5_dl5_self_n%s.json' %(noise_level): 'unlensed'}
#flist_dict = {'results/F_mat_CDMp_prior_lensys_delensed_scalar_bin5_dl5_self_n%s.json' %(noise_level): 'delensed'}
'''
flist_dict = {
#'results/20220930_no_lensing_sys/F_mat_CDMp_test_total_bin5_dl5_self_n%.1f.json' %(noise_level): 'lensed',
'results/F_mat_CDMp_test_delensed_scalar_bin5_dl5_self_n%.1f.json' %(noise_level): 'delensed',
#'results/20220930_no_lensing_sys/F_mat_CDMp_test_unlensed_total_bin5_dl5_self_n%.1f.json' %(noise_level): 'unlensed',
}


flist_dict_values = list(flist_dict.values())


#fiducial parameter values
paramfile = 'params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt'
param_dict_ori = tools.get_ini_param_dict(paramfile)

#read the Fisher matrices and push into a dictionary
fix_params = ['r']
fix_params = []#'alpha_phi_sys']#, 'ombh2']#, 'omch2']#, 'thetastar']
#fix_params = ['ombh2', 'omch2', 'thetastar']
#fix_params = ['As', 'alpha_phi_sys', 'ombh2', 'omch2', 'thetastar', 'r', 'ns', 'tau']
#fix_params = ['r', 'ombh2', 'thetastar', 'tau']
prior_dic = {}
if tau_prior is not None:
    prior_dic['tau']=tau_prior
    prior_dic['alpha_phi_sys']=0.0001
    prior_dic['A_phi_sys']=1e-23
F_dict = {}
which_spectra_arr = []
for fname in flist_dict:
    which_spectra = flist_dict[fname]
    curr_dict = json.load( open(fname) )
    param_names = curr_dict['parms']
    F_mat = np.asarray( curr_dict['Fmat'] )
    if reqd_fsky is not None: 
        F_mat = F_mat / curr_dict['fsky'] * reqd_fsky
    F_mat, param_names = tools.fix_params(F_mat, param_names, fix_params)
    ###print(param_names); sys.exit()
    F_mat = tools.add_prior(F_mat, param_names, prior_dic)
    F_dict[which_spectra] = F_mat
    which_spectra_arr.append( which_spectra )
    #print( fname, param_names, curr_dict['fsky'], curr_dict.keys() )

#make corner plot
color_dict = {'lensed': ['navy'], 'delensed': ['darkgreen'], 'unlensed': ['red']}
#color_dict = {0: ['darkgreen'], 1: ['goldenrod'], 2: ['darkred']}
fsval = 11
lwval = 1.25

#get fiducial and delta values for plotting
param_delta_dict = {'As': 2e-11, 'ns': .002, 'ombh2': 5e-5, 'omch2': 1e-3, 'tau': 0.01, 'thetastar': 1e-6, 'r': 0.001, 'A_phi_sys': 1e-19, 'alpha_phi_sys': 1.}
param_dict = {}
for p in param_names:
    param_dict[p] = [param_dict_ori[p], param_delta_dict[p]]
    print(p, param_dict[p])
##sys.exit()
 
desired_params_to_plot = param_names #params that must be plotted. Can be a subset of param_names or just param_names (if you want to plot all of them).
#desired_params_to_plot = ['A_phi_sys', 'As', 'alpha_phi_sys', 'ns', 'r', 'tau']
#desired_params_to_plot = ['A_phi_sys', 'alpha_phi_sys', 'As', 'ombh2', 'ns']
#print(desired_params_to_plot); sys.exit()
figlen = len(desired_params_to_plot) + 4
clf()
figure(figsize=(figlen, figlen))
fix_axis_range_to_xxsigma = 4. ##None
tools_for_plotting.make_triangle_plot(which_spectra_arr, F_dict, param_dict, param_names, desired_params_to_plot, color_dict, fsval = fsval, lwval = lwval, fix_axis_range_to_xxsigma = fix_axis_range_to_xxsigma)#, upper_or_lower_triangle = 'upper')

if (1): #add legend
    tr = tc = len(desired_params_to_plot)
    ax = subplot(tr, tc, tr+4)
    for which_spectra in color_dict:
        if which_spectra not in flist_dict_values: continue
        plot([], [], ls = '-', color = color_dict[which_spectra][0], label = r'%s' %(which_spectra.capitalize()), lw = lwval)
    plot([], [], ls = '-', color = 'white', label = r'Noise level: $\Delta_{\rm T} = %s \mu$K-arcmin' %(noise_level) )
    for p in prior_dic:
        plot([], [], ls = '-', color = 'white', label = r'Prior: $\sigma$(%s) = %g' %(tools_for_plotting.get_latex_param_str(p), prior_dic[p]) )

    
    #legend(loc = 3, fontsize = fsval + 6, framealpha = 0., handlelength=0.)
    legend(loc = 3, fontsize = fsval + 6, framealpha = 0.)

    axis('off')

param_names_str = '-'.join(desired_params_to_plot)
plfolder = 'plots/'
if not os.path.exists(plfolder): os.system('mkdir -p %s' %(plfolder))
plname = '%s/constraints_%s_noiselevel%s' %(plfolder, param_names_str, noise_level)
if len(prior_dic) >= 1: #
    prior_str = ''
    for keyname in prior_dic:
        prior_str = '%s-%s%s' %(prior_str, keyname, prior_dic[keyname])
    prior_str = prior_str.strip('-')
    plname = '%s_%s' %(plname, prior_str)
if reqd_fsky is not None:
    plname = '%s_fsky%s' %(plname, reqd_fsky)
plname = '%s.png' %(plname)
print(plname)
savefig(plname, dpi = 300.)
#show(); 
sys.exit()
