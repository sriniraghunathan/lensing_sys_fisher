import numpy as np, sys, scipy as sc, os
import json

from scipy import linalg
import copy
from scipy import interpolate 
from scipy.interpolate import interp1d
from itertools import combinations

########################################################################################################################
def write_log(logline, log_file = None):
    if log_file is not None:
        logfile = open(log_file, 'a')
        log_file.writelines('%s\n' %(logline))
    print(logline)
########################################################################################################################
def get_ini_param_dict(fpath = 'params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt'):
    """
    read params file and initialise cosmology
    """
    try:
        params = np.recfromtxt(fpath, delimiter = '=', encoding = 'utf-8')
    except:
        params = np.recfromtxt(fpath, delimiter = '=')
    param_dict = {}
    for rec in params:
        val = rec[1].strip()##.decode("utf-8")
        try:
            val = val.decode("utf-8")
        except:
            pass
        try:
            if val.find('.')>-1:
                val = float(val)
            else:
                val = int(val)
        except:
            val = str(val)

        if val == 'None':
            val = None
        paramname = rec[0].strip()
        try:
            paramname = paramname.decode("utf-8")
        except:
            pass
        param_dict[paramname] = val

    return param_dict


########################################################################################################################
def get_ini_cmb_power(param_dict, raw_cl = 1):

    """
    set CAMB cosmology and get power spectra
    """

    import camb
    from camb import model, initialpower
    from camb import correlations
    
    param_dict_to_use = copy.deepcopy(param_dict)

    ########################
    #set all CAMB parameters
    #pars = camb.CAMBparams()
    pars = camb.CAMBparams(max_l_tensor = param_dict_to_use['max_l_tensor'], max_eta_k_tensor = param_dict_to_use['max_eta_k_tensor'])
    pars.set_accuracy(AccuracyBoost = param_dict_to_use['AccuracyBoost'], lAccuracyBoost = param_dict_to_use['lAccuracyBoost'], 
        lSampleBoost = param_dict_to_use['lSampleBoost'],
        DoLateRadTruncation = param_dict_to_use['do_late_rad_truncation'])
    pars.set_dark_energy(param_dict_to_use['ws'])
    pars.set_cosmology(thetastar=param_dict_to_use['thetastar'], ombh2=param_dict_to_use['ombh2'], omch2=param_dict_to_use['omch2'], nnu = param_dict_to_use['neff'], 
        mnu=param_dict_to_use['mnu'], omk=param_dict_to_use['omk'], tau=param_dict_to_use['tau'], YHe = param_dict_to_use['YHe'], Alens = param_dict_to_use['Alens'],
        num_massive_neutrinos = param_dict_to_use['num_nu_massive'])
    pars.set_for_lmax(int(param_dict_to_use['max_l_limit']), lens_potential_accuracy=param_dict_to_use['lens_potential_accuracy'],
        max_eta_k = param_dict_to_use['max_eta_k'])
    pars.InitPower.set_params(ns=param_dict_to_use['ns'], r=param_dict_to_use['r'], As = param_dict_to_use['As'])
    ########################

    ########################
    #get results
    pars.WantTensors = True
    results = camb.get_results(pars)
    ########################

    ########################
    #get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = raw_cl, CMB_unit='muK')#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
    ########################

    ########################
    #get only the required ell range since powerspectra start from ell=0 by default
    for keyname in powers:
        powers[keyname] = powers[keyname][param_dict['min_l_limit']:, :]
    els = np.arange(param_dict['min_l_limit'], param_dict['max_l_limit']+1)
    ########################


    if param_dict['uK']:
        #powers['total'] *= 1e12
        #powers['unlensed_total'] *= 1e12
        cl_phiphi, cl_Tphi, cl_Ephi = powers['lens_potential'].T
        cphifun = interpolate.interp1d(els, cl_phiphi)
        totCL=powers['total']
        unlensedCL=powers['unlensed_total']

        bl, nlT, nlP = get_nl(els, 2., None, 1.)

    return els, powers
    #return els

########################################################################################################################
def get_cmb_spectra_using_camb(param_dict, which_spectra, step_size_dict_for_derivatives = None, raw_cl = 1, high_low = 0, verbose = True, debug = False, noise_nzero_fname = None):

    """
    set CAMB cosmology and get power spectra
    """

    import camb
    from camb import model, initialpower
    from camb import correlations
    
    ########
    #modify parameter values using step_size_dict_for_derivatives to compute derivatives using finite difference method
    if step_size_dict_for_derivatives is not None:
        param_dict_mod = param_dict.copy()

        for keyname in step_size_dict_for_derivatives.keys():
            print('keyname: ', keyname)
            print('keyname param: ', param_dict_mod[keyname])
            print(' step_size_dict_for_derivatives ', step_size_dict_for_derivatives)
            print(' step_size_dict_for_derivatives value', step_size_dict_for_derivatives[keyname])
            if high_low == 0:
                logline = '\t\tModifying %s for derivative now: (%s + %s)' %(keyname, param_dict_mod[keyname], step_size_dict_for_derivatives[keyname])
                param_dict_mod[keyname] = param_dict_mod[keyname] + step_size_dict_for_derivatives[keyname]
                
            else:
                logline = '\t\tModifying %s for derivative now: (%s - %s)' %(keyname, param_dict_mod[keyname], step_size_dict_for_derivatives[keyname])
                param_dict_mod[keyname] = param_dict_mod[keyname] - step_size_dict_for_derivatives[keyname]
            if verbose: write_log(logline)
        param_dict_to_use = copy.deepcopy(param_dict_mod)
    else:
        param_dict_to_use = copy.deepcopy(param_dict)

    ########################
    #set all CAMB parameters
    #pars = camb.CAMBparams()
    pars = camb.CAMBparams(max_l_tensor = param_dict_to_use['max_l_tensor'], max_eta_k_tensor = param_dict_to_use['max_eta_k_tensor'])
    pars.set_accuracy(AccuracyBoost = param_dict_to_use['AccuracyBoost'], lAccuracyBoost = param_dict_to_use['lAccuracyBoost'], 
        lSampleBoost = param_dict_to_use['lSampleBoost'],
        DoLateRadTruncation = param_dict_to_use['do_late_rad_truncation'])
    pars.set_dark_energy(param_dict_to_use['ws'])
    pars.set_cosmology(thetastar=param_dict_to_use['thetastar'], ombh2=param_dict_to_use['ombh2'], omch2=param_dict_to_use['omch2'], nnu = param_dict_to_use['neff'], 
        mnu=param_dict_to_use['mnu'], omk=param_dict_to_use['omk'], tau=param_dict_to_use['tau'], YHe = param_dict_to_use['YHe'], Alens = param_dict_to_use['Alens'],
        num_massive_neutrinos = param_dict_to_use['num_nu_massive'])
    pars.set_for_lmax(int(param_dict_to_use['max_l_limit']), lens_potential_accuracy=param_dict_to_use['lens_potential_accuracy'],
        max_eta_k = param_dict_to_use['max_eta_k'])
    #print(param_dict_to_use); sys.exit()
    pars.InitPower.set_params(ns=param_dict_to_use['ns'], r=param_dict_to_use['r'], As = param_dict_to_use['As'])
    ########################

    ########################
    #get results
    pars.WantTensors = True
    results = camb.get_results(pars)
    ########################

    ########################
    #get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = raw_cl, CMB_unit='muK')#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
    ########################

    ########################
    #get only the required ell range since powerspectra start from ell=0 by default
    for keyname in powers:
        powers[keyname] = powers[keyname][param_dict['min_l_limit']:, :]
    els = np.arange(param_dict['min_l_limit'], param_dict['max_l_limit']+1)
    ########################

    if debug:
        import matplotlib.pyplot as plt
        #from IPython import embed; embed()
        unlensed, lensed, phi = powers['unlensed_scalar'], powers['lensed_scalar'], powers['lens_potential']
        total = powers['total']
        #unlensed_v2, lensed_v2, phi_v2 = powers['unlensed_scalar'], powers['lensed_scalar'], powers['lens_potential']

        #powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = 0)#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
        #unlensed, lensed, phi = powers['unlensed_scalar'], powers['lensed_scalar'], powers['lens_potential']

        ax = plt.subplot(111, yscale = 'log');plt.plot(unlensed[:,0], 'k-', label = r'Unlensed'); # plot(unlensed_v2[:,0], 'lime'); show()
        ax = plt.subplot(111, yscale = 'log');plt.plot(lensed[:,0], 'g-', label = r'Lensed'); #plot(lensed_v2[:,0]/lensed[:,0]); show()
        #ax = plt.subplot(111, yscale = 'log');plt.plot(total[:,0], 'r-', label = r'Unlensed'); #plot(lensed_v2[:,0]/lensed[:,0]); show()
        plt.legend(loc = 1)
        plt.xlim(0, 5000); plt.ylim(1e-6, 500.)
        plt.show()
        ax = plt.subplot(111, yscale = 'log'); plt.plot(phi[:,0], 'k-'); #plot(phi_v2[:,0], 'lime'); show()
        plt.xlim(0, 5000); plt.ylim(1e-23, 1e-9); 
        plt.show()
        #sys.exit()

    ########################
    #add delensedCL
    if which_spectra == 'delensed_scalar':
        #if param_dict['uK']:
         #   powers['total'] *= 1e12
          #  powers['unlensed_total'] *= 1e12
        cl_phiphi, cl_Tphi, cl_Ephi = powers['lens_potential'].T
        cphifun = interpolate.interp1d(els, cl_phiphi)
        totCL=powers['total']
        unlensedCL=powers['unlensed_total']
        print('lenels ',len(els))
        print('lenecl ',unlensedCL.shape)
        
        rmsT = param_dict['rms_map_T']
        rmsP = param_dict['rms_map_P']
        nlT = param_dict['nlT']
        nlP = param_dict['nlP']
        fwhm = param_dict['fwhm_arcmins']
        binsize = param_dict['binsize']
        A_phi_sys_value=param_dict_to_use['A_phi_sys']
        alpha_phi_sys_value=param_dict_to_use['alpha_phi_sys']
        #A_phi_sys = param_dict['A_phi_sys']
        #alpha_phi_sys = param_dict['alpha_phi_sys']
        ##output_name = "params/generate_n0s_rmsT%s_fwhmm%s.dat"%(rmsT, fwhm)
        if noise_nzero_fname is None:
            noise_nzero_fname = 'params/generate_n0s_rmsT%.1f_fwhmm%.1f_dl%d.dat'%(rmsT, 1.0, binsize)
        n0s = np.loadtxt(noise_nzero_fname)
        nels = n0s[:,0]
        mv = n0s[:,-1]
        '''
        bl, nlT, nlP = get_nl(els, 2., None, 1.)
        nels = np.arange(els[0], els[-1]+100, 100)
        n0s = calculate_n0(nels, els, unlensedCL, totCL, nlT, nlP, dimx = 1024, dimy = 1024, fftd = 1./60/180)
        mv = 1./(1./n0s['EB']+1./n0s['EE']+1./n0s['TT']+1./n0s['TB']+1./n0s['TE'])
        data = np.column_stack((nels,n0s['EB'],n0s['EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv))
        header = "els,n0s['EB'],n0s[q'EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv" 
        output_name = "params/generate_n0s.dat"
        np.savetxt(output_name, data, header=header)
        #rhosq = cphifun(els) / (cphifun(els) + n0fun(els) + Nadd(els))
        #rhosqfun = interpolate.interp1d(els, rhosq)
        '''
        n0fun = interpolate.interp1d(nels, mv)

        winf = cphifun(els) / (n0fun(els) + cphifun(els) + get_nl_sys(els, A_phi_sys_value, alpha_phi_sys_value))
        print('cphi: ',cphifun(els))
        print('n0: ', n0fun(els))
        print('nlsys: ', get_nl_sys(els, A_phi_sys_value, alpha_phi_sys_value))
        recov_cl_phiphi = cl_phiphi + get_nl_sys(els, A_phi_sys_value, alpha_phi_sys_value)
        clpp = cl_phiphi * (1.-winf**1) * (els*(els+1))**2/2/np.pi

        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        clpp = np.insert(clpp, 0, np.zeros(2), axis = 0)
        thyres = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        #thyrese = thyres[:,1]
        #thyrest = thyres[:,0]
        #thyresb = thyres[:,2]
        #thyreste = thyres[:,3]
        
        #test_a_param
        test_a_param = 0
        if not test_a_param:
            powers[which_spectra] = thyres[param_dict['min_l_limit']:, :]
            powers[which_spectra] = powers[which_spectra] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
        else:
            powers[which_spectra] = (unlensedCL + totCL)/2
        #powers[which_spectra] *= 1e-12

        '''
        delensed_dict,  delensedCL=get_delensed_from_lensed(nels, els, unlensedCL, totCL , cphifun, n0fun,  nlT, nlP, dimx = 1024, dimy = 1024, fftd = 1./60/180)
        #delensed_dict,  delensedCL=get_delensed_from_lensed_cvltion(els, unlensedCL, totCL , cphifun, rhosqfun,  nlT, nlP, dimx = 1024, dimy = 1024, fftd = 1./60/180)
        fundelt = interpolate.interp1d(nels, delensed_dict['TT'])
        fundele = interpolate.interp1d(nels, delensed_dict['EE'])
        fundelb = interpolate.interp1d(nels, delensed_dict['BB'])
        fundelte = interpolate.interp1d(nels, delensed_dict['TE'])
        data = np.column_stack(( totCL[:,0] - fundelt(els), totCL[:,1]-fundele(els), fundelb(els), totCL[:,3] - fundelte(els) ))
        powers[which_spectra] = data
        powers[which_spectra] *= 1e-12
        '''
    ########################



    ########################
    #do we need cl or dl
    if not raw_cl: #20200529: also valid for lensing (see https://camb.readthedocs.io/en/latest/_modules/camb/results.html#CAMBdata.get_lens_potential_cls)
        powers[which_spectra] = powers[which_spectra] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
    ########################

    ########################    
    #Tcmb factor
    if pars.OutputNormalization == 1:
        powers[which_spectra] = param_dict['T_cmb']**0. *  powers[which_spectra]
        #powers[which_spectra] = param_dict['T_cmb']**2. *  powers[which_spectra]
    ########################

    ########################
    #K or uK
    #if param_dict['uK']:
        #powers[which_spectra] *= 1e12
    ########################        

    cl_tt, cl_ee, cl_bb, cl_te = powers[which_spectra].T
    #print(cl_tt.max(), param_dict_to_use['As'])

    cl_dict = {}
    cl_dict['TT'] = cl_tt
    cl_dict['EE'] = cl_ee
    cl_dict['BB'] = cl_bb
    cl_dict['TE'] = cl_te

    cl_phiphi, cl_Tphi, cl_Ephi = powers['lens_potential'].T

    #K or uK
    if param_dict['uK']:
        cl_Tphi *= 1e6##1e12
        cl_Ephi *= 1e6##1e12

    cl_phiphi = cl_phiphi# * (els * (els+1))**2. /(2. * np.pi)
    cl_Tphi = cl_Tphi# * (els * (els+1))**1.5 /(2. * np.pi)
    cl_Ephi = cl_Ephi# * (els * (els+1))**1.5 /(2. * np.pi)
    
    cl_dict['PP'] = cl_phiphi
    cl_dict['Tphi'] = cl_Tphi
    cl_dict['Ephi'] = cl_Ephi

    return pars, els, cl_dict

########################################################################################################################


########################################################################################################################

def get_derivatives(param_dict, which_spectra, step_size_dict_for_derivatives = None, params_to_constrain = None, noise_nzero_fname = None):

    """
    compute derivatives using finite difference method
    """
    if step_size_dict_for_derivatives is None:
        step_size_dict_for_derivatives = get_step_sizes_for_derivative_calc(params_to_constrain)

    cl_deriv = {}
    for keyname in sorted(step_size_dict_for_derivatives):
        if params_to_constrain is not None and keyname not in params_to_constrain: continue
        tmpdic = {}
        tmpdic[keyname] = step_size_dict_for_derivatives[keyname]

        #compute power with fid+step
        dummypars, els, cl_mod_dic_1 = get_cmb_spectra_using_camb(param_dict, which_spectra, step_size_dict_for_derivatives = tmpdic, high_low = 0, noise_nzero_fname = noise_nzero_fname)
        #compute power with fid-step
        dummypars, els, cl_mod_dic_2 = get_cmb_spectra_using_camb(param_dict, which_spectra, step_size_dict_for_derivatives = tmpdic, high_low = 1, noise_nzero_fname = noise_nzero_fname)

        #get derivative using finite difference method
        cl_deriv[keyname] = {}
        for XX in cl_mod_dic_2: #loop over TT,EE,BB,TE
            cl_deriv[keyname][XX] = (cl_mod_dic_1[XX] - cl_mod_dic_2[XX]) / (2*tmpdic[keyname])

    return cl_deriv

########################################################################################################################
def get_step_sizes_for_derivative_calc(params_to_constrain):
    """
    #step sizes for derivatives. Generally soemthing close to (parameter * 0.001)
    """
    step_size_dict_for_derivatives = {\
    'ombh2' : 0.0008,
    'omch2' : 0.0030,
    'tau' : 0.020,
    ###'tau':0.0002,
    ###'As': 0.1e-11,
    'As': 0.1e-9,
    'ns' : 0.010,
    ###'ws' : -1e-2,
    ###'neff': 0.080,
    'neff': 0.0080,
    ###'mnu': 0.02,
    'mnu':0.0006,
    'A_phi_sys': 1e-20,
    'alpha_phi_sys': -2e-2,
    'r': 0.0001,
    ###'YHe': 0.005, 
    #'Alens': 1e-2, 
    #'Aphiphi': 1e-2, 
    'thetastar': 0.000050, 
    }
    ref_step_size_dict_for_derivatives = {}
    for p in step_size_dict_for_derivatives:
        if p not in params_to_constrain: continue
        ref_step_size_dict_for_derivatives[p] = step_size_dict_for_derivatives[p]
    return ref_step_size_dict_for_derivatives


########################################################################################################################
def get_derivative_to_phi_with_camb(els, which_spectra, unlensedCL, cl_phiphi, nl_dict, Ls_to_get, percent=0.05):
    """
    #CAlculate dC^xx/dC^phiphi
    """
    diff_dict = {}
    import camb
    from camb import model, initialpower
    from camb import correlations

    if which_spectra == "unlensed_scalar":
        diff_dict = {}

    elif which_spectra == "lensed_scalar":
        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        clpp = np.insert(cl_phiphi, 0, np.zeros(2), axis = 0)
        thyres0 = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        diff = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        for i, Li in enumerate(Ls_to_get):
            clpp[Li] = clpp[Li]*(1+percent)
            thyresi = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
            diff[i] = (thyresi - thyres0)/(clpp[Li]*percent)
        diff_dict['TT'] = diff[:, 2:, 0] * 2 * np.pi / (els * (els + 1 ))
        diff_dict['EE'] = diff[:, 2:, 1] * 2 * np.pi / (els * (els + 1 ))
        diff_dict['TE'] = diff[:, 2:, 3] * 2 * np.pi / (els * (els + 1 ))

    elif which_spectra == "delensed_scalar":
        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        winf0 = cl_phiphi / (nl_dict['PP'] + cl_phiphi + nl_dict['SYS'])
        recov_cl_phiphi = cl_phiphi + nl_dict['SYS']
        clpp = cl_phiphi * (1.-winf0**1) * (els*(els+1))**2/2/np.pi        
        clpp = np.insert(clpp, 0, np.zeros(2), axis = 0)
        thyres0 = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        diff = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        for i, Li in enumerate(Ls_to_get):
            clpp[Li-2] = cl_phiphi[Li-2]*(1+percent)            
            winf = cl_phiphi / (nl_dict['PP'] + cl_phiphi + nl_dict['SYS'])
            recov_cl_phiphi = cl_phiphi + nl_dict['SYS']
            clpp = cl_phiphi * (1.-winf**1) * (els*(els+1))**2/2/np.pi
            clpp = np.insert(clpp, 0, np.zeros(2), axis = 0)
            thyresi = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
            diff[i] = (thyresi - thyres0)/(clpp[Li]*percent)
        
        diff_dict['TT'] = diff[:, 2:, 0] * 2 * np.pi / (els * (els + 1 ))
        diff_dict['EE'] = diff[:, 2:, 1] * 2 * np.pi / (els * (els + 1 ))
        diff_dict['TE'] = diff[:, 2:, 3] * 2 * np.pi / (els * (els + 1 ))
    return diff_dict

########################################################################################################################
def get_nl_dict(nlfile, els):
    """
    #Obtain formatted nl_dict from an ILC residual file
    """
    res_dict  = np.load(nlfile, allow_pickle = 1, encoding = 'latin1').item()
    el_nl, cl_residual = res_dict['el'], res_dict['cl_residual']

    if 'T' in cl_residual:
        nl_tt, nl_ee = cl_residual['T'], cl_residual['P']
        nl_tt = np.interp(els, el_nl, nl_tt)
        nl_ee = np.interp(els, el_nl, nl_ee)
        nl_te = None
    else:
        nl_tt, nl_ee = cl_residual['TT'], cl_residual['EE']
        if 'TE' in cl_residual:
            nl_te = cl_residual['TE']
        else:
            nl_te = None
        el_nl = np.arange(len(nl_tt))
        nl_tt = np.interp(els, el_nl, nl_tt)
        nl_ee = np.interp(els, el_nl, nl_ee)
        if nl_te is not None:
            nl_te = np.interp(els, el_nl, nl_te)
        else:
            nl_te = np.zeros(len(els))

    nl_dict  = {}
    nl_dict['TT'] = nl_tt
    nl_dict['EE'] = nl_ee
    nl_dict['TE'] = nl_te

    return nl_dict
########################################################################################################################

def get_delta_cl(els, cl_dict, nl_dict, fsky = 1., include_lensing = True):
    """
    get Delta_cl (sample variance)
    """
    delta_cl_dict  = {}
    for XX in cl_dict:
        if XX == 'TT':
            nl = nl_dict['TT']
        elif XX == 'EE' or XX == 'BB':
            nl = nl_dict['EE']
        elif XX == 'TE':
            nl = nl_dict['TE']
            nl = np.copy(nl) * 0.
        elif XX == 'PP' and include_lensing:
            nl = nl_dict['PP']
        cl = cl_dict[XX]
        if XX == '0':
            delta_cl_dict[XX] = np.sqrt(2./ (2.*els + 1.) / fsky ) * ((cl_dict['TT'] + nl_dict['TT'])*(cl_dict['EE'] + nl_dict['EE'])+cl_dict['TE']**2)**0.5
        else:
            delta_cl_dict[XX] = np.sqrt(2./ (2.*els + 1.) / fsky ) * (cl + nl)

    return delta_cl_dict

########################################################################################################################

def get_delta_cl_cov(els, cl_dict, nl_dict, fsky = 1., include_lensing = False, include_B = False, dB_dE_dict = None, diff_phi_dict = None, diff_self_dict = None, which_spectra = None, Ls_to_get = None):
    """
    get Delta_cl (sample variance)
    """
    clname = ['TT','EE','TE']
    nl_dict['ET'] = nl_dict['TE']
    cl_dict['ET'] = cl_dict['TE']
    #if include_lensing:
    #clname = ['TT','EE','TE','PP']
    #comb2 = list(combinations(clname, 2))

    cov_dict = {}
    for i, namei in enumerate(clname):
        for j, namej in enumerate(clname):
            pair1 = namei[0]+namej[0]
            pair2 = namei[1]+namej[1]
            pair3 = namei[0]+namej[1]
            pair4 = namei[1]+namej[0]
            totname = namei + namej
            covij = 1/fsky / (2.*els + 1.) * ( (cl_dict[pair1]+nl_dict[pair1])*(cl_dict[pair2]+nl_dict[pair2]) + (cl_dict[pair3]+nl_dict[pair3])*(cl_dict[pair4]+nl_dict[pair4]) )
            cov_dict[totname] = covij
    
    cov_dict['PPPP'] = 2 /fsky / (2.*els + 1.) * (cl_dict['PP'] + nl_dict['PP'])**2
    
    if include_B:
        print("Include BB")
        term1 = 2/fsky / (2.*els + 1.) * ( (cl_dict['BB']+nl_dict['BB'])*(cl_dict['BB']+nl_dict['BB']))
        print('term1: ', term1)
        if which_spectra == "unlensed_total":
            cov_dict['BBBB'] = np.diag(term1)
            for item in clname:
                cov_dict['BB'+item] = np.zeros(cov_dict['BBBB'].shape)
                print(cov_dict.keys())
        
        if which_spectra == "total" or which_spectra == "delensed_scalar":
            cov_dict['BBBB'] = np.diag(term1)
            for item in clname:
                cov_dict['BB'+item] = np.zeros(cov_dict['BBBB'].shape)

            print("which_spectra is %s"%(which_spectra))
            if include_lensing:
                deriv_filter = np.zeros(dB_dE_dict['BB'].shape)
                deriv_filter[8:, 30:] = 1
                sumle = np.einsum('ai,a,aj->ij', dB_dE_dict['BB']*deriv_filter, cov_dict['EEEE'][Ls_to_get-2], dB_dE_dict['BB']*deriv_filter)
                sumlp = np.einsum('ai,a,aj->ij', diff_phi_dict['BB']*deriv_filter, cov_dict['PPPP'][Ls_to_get-2], diff_phi_dict['BB']*deriv_filter)
                dl = Ls_to_get[1] - Ls_to_get[0]
                cov_dict['BBBB'] = cov_dict['BBBB'] + sumle*dl + sumlp*dl
                print('BBBB: ', cov_dict['BBBB'])
                dxylen_dxyulen = diff_self_dict
                #dxylen_dxyulen[Ls_to_get-1] = 1
                #dxylen_dxyulen = np.fll_diagonal(dxylen_dxyulen, 1)
                for item in clname:
                    sumle = np.einsum('ai,a,aj->ij', dB_dE_dict['BB']*deriv_filter, cov_dict['EE'+item][Ls_to_get-2], dxylen_dxyulen[item]*deriv_filter)
                    sumlp = np.einsum('ai,a,aj->ij', diff_phi_dict['BB']*deriv_filter, cov_dict['PPPP'][Ls_to_get-2], diff_phi_dict[item]*deriv_filter)
                    cov_dict['BB'+item] = sumlp*dl + sumle*dl
                    #cov_dict['BB'+item] = np.zeros((cov_dict['BBBB'].shape))
                    print(cov_dict.keys())

        print('BBBB: ', cov_dict['BBBB'])
    
    if include_lensing:
        if which_spectra == "total" or which_spectra == "delensed_scalar":
            for keyi in cov_dict:
                if keyi != "PPPP" and keyi[0] != "B":
                    sumlp = np.einsum('ai,a,aj->ij', diff_phi_dict[keyi[0]+keyi[1]]*deriv_filter, cov_dict['PPPP'][Ls_to_get-2], diff_phi_dict[keyi[2]+keyi[3]]*deriv_filter)   
                    if len(cov_dict[keyi].shape) == 2:
                        cov_dict[keyi] += sumlp*dl
                    else:
                        cov_dict[keyi] = np.diag(cov_dict[keyi]) + sumlp*dl

    return cov_dict


########################################################################################################################

def get_delta_cl_cov2(els, cl_dict, nl_dict, fsky = 1., include_lensing = False, include_B = False, dB_dE_dict = None, diff_phi_dict = None, diff_self_dict = None, which_spectra = None, Ls_to_get = None):
    """
    get Delta_cl (sample variance)
    """
    nl_dict['ET'] = nl_dict['TE']
    cl_dict['ET'] = cl_dict['TE']
    #if include_lensing:
    #clname = ['TT','EE','TE','PP']
    #comb2 = list(combinations(clname, 2))
    clname = ['TT','EE','TE']

    cov_dict = {}
    for i, wx in enumerate(clname):
        for j, yz in enumerate(clname):
            xy = wx[1] + yz[0]
            wz = wx[0] + yz[1]
            pair1 = xy[0]+wz[0]
            pair2 = xy[1]+wz[1]
            pair3 = xy[0]+wz[1]
            pair4 = xy[1]+wz[0]
            totname = xy + wz
            covij = 1/fsky / (2.*els + 1.) * ( (cl_dict[pair1]+nl_dict[pair1])*(cl_dict[pair2]+nl_dict[pair2]) + (cl_dict[pair3]+nl_dict[pair3])*(cl_dict[pair4]+nl_dict[pair4]) )
            cov_dict[totname] = covij


    
    if include_B:
        temp_cov = cov_dict.copy()
        temp_cov['EETT'] = 2/fsky / (2.*els + 1.) * (cl_dict['TE']+nl_dict['TE'])**2
        for item in cov_dict.keys():
            cov_dict[item] = np.diag(cov_dict[item])
        cov_dict['PPPP'] = 2 /fsky / (2.*els + 1.) * (cl_dict['PP'] + nl_dict['PP'])**2
        term1 = 2/fsky / (2.*els + 1.) * ( (cl_dict['BB']+nl_dict['BB'])*(cl_dict['BB']+nl_dict['BB']))**2
        if which_spectra == "unlensed_scalar":
            cov_dict['BBBB'] = np.diag(term1)
            for item in clname:
                cov_dict['BB'+item] = 0
                cov_dict[item+'BB'] = 0
        
        if which_spectra == "total" or which_spectra == "delensed_scalar":
            sumle = np.einsum('ai,a,aj->ij', dB_dE_dict['BB'], np.diag(cov_dict['EEEE'])[Ls_to_get-2], dB_dE_dict['BB'])
            sumlp = np.einsum('ai,a,aj->ij', diff_phi_dict['BB'], cov_dict['PPPP'][Ls_to_get-2], diff_phi_dict['BB'])
            cov_dict['BBBB'] = np.diag(term1) + sumle + sumlp
            dxylen_dxyulen = diff_self_dict
            #dxylen_dxyulen[Ls_to_get-1] = 1
            #dxylen_dxyulen = np.fll_diagonal(dxylen_dxyulen, 1)
            for item in clname:
                sumle = np.einsum('ai,a,aj->ij', dB_dE_dict['BB'], temp_cov['EE'+item][Ls_to_get-2], dxylen_dxyulen[item])
                sumlp = np.einsum('ai,a,aj->ij', diff_phi_dict['BB'], np.diag(cov_dict['EEEE'])[Ls_to_get-2], diff_phi_dict[item])
                cov_dict['BB'+item] = sumlp + sumle
                cov_dict[item+'BB'] = sumlp + sumle

    return cov_dict


########################################################################################################################

def get_nongaussaian_cl_cov(which_spectra, Ls_to_get, diffphi, delta_cl_dict, els, cl_phiphi, nl_dict, fsky = 1.):
    """
    get non_gaussian part for Delta_cl (sample variance)
    """
    cov_nongaussian = {}
    phi_at_ls_total = cl_phiphi[Ls_to_get-2] + nl_dict['PP'][Ls_to_get-2] * 2/fsky / (2.*els[Ls_to_get-2] + 1)
    
    dl = Ls_to_get[1] - Ls_to_get[0]
    if which_spectra == "unlensed_scalar":
        cov_nongaussian = {}
    else:
        clname = ['TT','EE','TE']
        for i, namei in enumerate(clname):
            for j, namej in enumerate(clname):
                totname = namei + namej
                covij = np.einsum('ai,a,aj->ij', diffphi[namei], phi_at_ls_total**2 , diffphi[namej])
                cov_nongaussian[totname] = covij*dl

    return cov_nongaussian

########################################################################################################################

def get_deriv_clBB(which_spectra, els, unlensedCL, cl_phiphi, nl_dict, Ls_to_get = np.arange(2, 5000, 100), percent=0.05, noiseTi  = 1):
    """
    get non_gaussian part for Delta_cl (sample variance)
    """
    import camb
    from camb import model, initialpower
    from camb import correlations

    diff_phi_dict = {}
    diff_EE_dict = {}
    diff_self_dict = {}
    dl = Ls_to_get[1] - Ls_to_get[0]

    if which_spectra == "total":
        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        clpp = np.insert(cl_phiphi*(els*(els+1))**2/2/np.pi, 0, np.zeros(2), axis = 0)
        #cl_phiphilong = np.insert(cl_phiphi, 0, np.zeros(2), axis = 0)
        thyres0 = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        diffp = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        diffe = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        diffs = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        for i, Li in enumerate(Ls_to_get):
            clppi = clpp.copy()
            clppi[Li] = clpp[Li]*(1+percent)
            thyresi = camb.correlations.lensed_cls(cls, clppi, lmax = els[-1])
            clsi = cls.copy()
            clsi[Li,1] = clsi[Li,1]*(1+percent)
            thyresj = camb.correlations.lensed_cls(clsi, clpp, lmax = els[-1])
            diffp[i] = (thyresi - thyres0)/(clpp[Li]*percent/(Li*(Li+1))**2)
            diffe[i] = (thyresj - thyres0)/(cls[Li,1]*percent/(Li*(Li+1)))
            diffs[i,:,1] = (thyresj[:,1] - thyres0[:,1])/(cls[Li,1]*percent/(Li*(Li+1)))
            clsi = cls.copy()
            clsi[Li,0] = clsi[Li,0]*(1+percent)
            thyresj = camb.correlations.lensed_cls(clsi, clpp, lmax = els[-1])
            diffs[i,:,0] = (thyresj[:,0] - thyres0[:,0])/(cls[Li,0]*percent/(Li*(Li+1))) ## second bug found
            clsi = cls.copy()
            clsi[Li,3] = clsi[Li,3]*(1+percent)
            thyresj = camb.correlations.lensed_cls(clsi, clpp, lmax = els[-1])
            diffs[i,:,3] = (thyresj[:,3] - thyres0[:,3])/(cls[Li,3]*percent/(Li*(Li+1)))

        diff_EE_dict['BB'] = diffe[:,2:,2] / (els * (els + 1 )) ## first bug found in BB
        diff_phi_dict['TT'] = diffp[:,2:,0] / (els * (els + 1 )) 
        diff_phi_dict['EE'] = diffp[:,2:,1] / (els * (els + 1 ))
        diff_phi_dict['BB'] = diffp[:,2:,2] / (els * (els + 1 ))
        diff_phi_dict['TE'] = diffp[:,2:,3] / (els * (els + 1 ))
        diff_self_dict['TT'] = diffs[:,2:,0] / (els * (els + 1 ))
        diff_self_dict['EE'] = diffs[:,2:,1] / (els * (els + 1 ))
        diff_self_dict['TE'] = diffs[:,2:,3] / (els * (els + 1 ))

        deriv_fname = 'derivs/diffphi_dl%s_Dl.json'%(dl)
        deriv_self_fname = 'derivs/diffself_dl%s_Dl.json' %(dl)
        deriv_ee_fname = 'derivs/diffee_dl%s_Dl.json' %(dl)

        with open(deriv_fname, 'w') as fp:
            j = json.dump({k: v.tolist() for k, v in diff_phi_dict.items()}, fp)
        with open(deriv_self_fname, 'w') as fp:
            j = json.dump({k: v.tolist() for k, v in diff_self_dict.items()}, fp)
        with open(deriv_ee_fname, 'w') as fp:
            j = json.dump({'BB': diff_EE_dict['BB'].tolist()}, fp)
        dataLs = np.column_stack((Ls_to_get))
        header0 = "Ls_to_get" 
        output_name0 = "derivs/Ls_to_get.dat"%(dl)
        np.savetxt(output_name0, dataLs, header=header0)

    elif which_spectra == "delensed_scalar":
        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        winf0 = cl_phiphi / (nl_dict['PP'] + cl_phiphi + nl_dict['SYS'])
        recov_cl_phiphi = cl_phiphi + nl_dict['SYS']
        clpp = cl_phiphi * (1.-winf0**1) * (els*(els+1))**2/2/np.pi
        clpp = np.insert(clpp, 0, np.zeros(2), axis = 0)
        thyres0 = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        diffp = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        diffe = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        diffs = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))
        for i, Li in enumerate(Ls_to_get):
            clphii = cl_phiphi.copy()
            clphii[Li-2] = cl_phiphi[Li-2]*(1+percent)
            winf = clphii / (nl_dict['PP'] + clphii + nl_dict['SYS'])
            recov_cl_phiphi = cl_phiphi + nl_dict['SYS']
            clppi = clphii * (1.-winf**1) * (els*(els+1))**2/2/np.pi
            clppi = np.insert(clppi, 0, np.zeros(2), axis = 0)
            thyresi = camb.correlations.lensed_cls(cls, clppi, lmax = els[-1])
            clsi = cls.copy()
            clsi[Li,1] = clsi[Li,1]*(1+percent)
            thyresj = camb.correlations.lensed_cls(clsi, clpp, lmax = els[-1])
            diffp[i] = (thyresi - thyres0)/(cl_phiphi[Li-2]*percent)
            diffe[i] = (thyresj - thyres0)/(clsi[Li, 1]*percent/(Li*(Li+1)))
            diffs[i,:,1] = (thyresj[:,1] - thyres0[:,1])/(clsi[Li,1]*percent/(Li*(Li+1)))
            clsi = cls.copy()
            clsi[Li,0] = clsi[Li,0]*(1+percent)
            thyresj = camb.correlations.lensed_cls(clsi, clpp, lmax = els[-1])
            diffs[i,:,0] = (thyresj[:,0] - thyres0[:,0])/(clsi[Li,0]*percent/(Li*(Li+1)))
            clsi = cls.copy()
            clsi[Li,3] = clsi[Li,3]*(1+percent)
            thyresj = camb.correlations.lensed_cls(clsi, clpp, lmax = els[-1])
            diffs[i,:,3] = (thyresj[:,3] - thyres0[:,3])/(clsi[Li,3]*percent/(Li*(Li+1)))

        diff_EE_dict['BB'] = diffe[:,2:, 2] / (els * (els + 1 ))
        diff_phi_dict['TT'] = diffp[:,2:,0]  / (els * (els + 1 ))
        diff_phi_dict['EE'] = diffp[:,2:,1]  / (els * (els + 1 ))
        diff_phi_dict['BB'] = diffp[:,2:,2]  / (els * (els + 1 ))
        diff_phi_dict['TE'] = diffp[:,2:,3]  / (els * (els + 1 ))
        diff_self_dict['TT'] = diffs[:,2:,0] / (els * (els + 1 ))
        diff_self_dict['EE'] = diffs[:,2:,1] / (els * (els + 1 ))
        diff_self_dict['TE'] = diffs[:,2:,3] / (els * (els + 1 ))

        deriv_fname = 'derivs/diffphi_dl%s_Dl_delensed_n%s.json' %(dl, noiseTi)
        deriv_self_fname = 'derivs/diffself_dl%s_Dl_delensed_n%s.json' %(dl, noiseTi)
        deriv_ee_fname = 'derivs/diffee_dl%s_Dl_delensed_n%s.json' %(dl, noiseTi)
    

        with open("derivs/diffphisys_dl%s_Dl_delensed_n%s.json"%(dl, noiseTi), 'w') as fp:
            j = json.dump({k: v.tolist() for k, v in diff_phi_dict.items()}, fp)
        with open("derivs/diffselfsys_dl%s_Dl_delensed_n%s.json"%(dl, noiseTi), 'w') as fp:
            j = json.dump({k: v.tolist() for k, v in diff_self_dict.items()}, fp)
        with open("derivs/diffeesys_dl%s_Dl_delensed_n%s.json"%(dl, noiseTi), 'w') as fp:
            j = json.dump({'BB': diff_EE_dict['BB'].tolist()}, fp)
        dataLs = np.column_stack((Ls_to_get))
        header0 = "Ls_to_get" 
        output_name0 = "derivs/Ls_to_get.dat"%(dl)
        np.savetxt(output_name0, dataLs, header=header0)

    
    return diff_EE_dict, diff_phi_dict, diff_self_dict

########################################################################################################################

def get_deriv_camb(which_spectra, els, unlensedCL, cl_phiphi, nl_dict, Ls_to_get = np.arange(2, 5000, 100), percent=0.05):
    """
    get non_gaussian part for Delta_cl (sample variance)
    """
    import camb
    from camb import model, initialpower
    from camb import correlations

    diff_phi_dict_full = {}
    diff_phi_dict = {}
    diff_EE_dict = {}
    diff_self_dict = {}
    dl = Ls_to_get[1] - Ls_to_get[0]
    dcambphi = 0
    dcambself = 0

    if which_spectra == "total":
        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        clpp = np.insert(cl_phiphi*(els*(els+1))**2/2/np.pi, 0, np.zeros(2), axis = 0)
        dcambself = camb.correlations.lensed_cl_derivative_unlensed(clpp, lmax=5000)
        dcambphi = camb.correlations.lensed_cl_derivatives(cls, clpp, lmax=5000)
        thyres0 = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        diffe = np.zeros((len(Ls_to_get), thyres0.shape[0], thyres0.shape[1]))

        for i, Li in enumerate(Ls_to_get):
            clppi = clpp.copy()
            clppi[Li] = clpp[Li]*(1+percent)
            thyresi = camb.correlations.lensed_cls(cls, clppi, lmax = els[-1])
            clsi = cls.copy()
            clsi[Li,1] = clsi[Li,1]*(1+percent)
            thyresj = camb.correlations.lensed_cls(clsi, clpp, lmax = els[-1])
            diffe[i] = (thyresj - thyres0)/(clsi[Li,1]*percent)

        diff_EE_dict['BB'] = diffe[:,2:,2] / (els * (els + 1 )) * els[Ls_to_get, np.newaxis] * (els[Ls_to_get, np.newaxis] +1 )
        #diff_phi_dict_full['BB'] = dcambphi[2,2:,:] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi
        #diff_phi_dict_full['TT'] = dcambphi[0,2:,:] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi
        #diff_phi_dict_full['TE'] = dcambphi[3,2:,:] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi
        #diff_phi_dict_full['EE'] = dcambphi[1,2:,:] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi
        diff_phi_dict['BB'] = dcambphi[2,2:,Ls_to_get] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi[Ls_to_get, np.newaxis]
        diff_phi_dict['TT'] = dcambphi[0,2:,Ls_to_get] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi[Ls_to_get, np.newaxis]
        diff_phi_dict['TE'] = dcambphi[3,2:,Ls_to_get] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi[Ls_to_get, np.newaxis]
        diff_phi_dict['EE'] = dcambphi[1,2:,Ls_to_get] * 2 * np.pi / (els * (els + 1 )) / cl_phiphi[Ls_to_get, np.newaxis]
        diff_self_dict['TT'] = dcambself[0,2:,Ls_to_get] / (els * (els + 1 )) * els[Ls_to_get, np.newaxis] * (els[Ls_to_get, np.newaxis] +1 )
        diff_self_dict['EE'] = dcambself[1,2:,Ls_to_get] / (els * (els + 1 )) * els[Ls_to_get, np.newaxis] * (els[Ls_to_get, np.newaxis] +1 )
        diff_self_dict['TE'] = dcambself[3,2:,Ls_to_get] / (els * (els + 1 )) * els[Ls_to_get, np.newaxis] * (els[Ls_to_get, np.newaxis] +1 )

        deriv_fname = 'derivs/diffphi_dl%s_camb.json'%(Lsdl)
        deriv_self_fname = 'derivs/diffself_dl%s_camb.json' %(Lsdl)
        deriv_ee_fname = 'derivs/diffee_dl%s_camb.json' %(Lsdl)

        with open(deriv_fname, 'w') as fp:
            j = json.dump({k: v.tolist() for k, v in diff_phi_dict.items()}, fp)
        with open(deriv_self_fname, 'w') as fp:
            j = json.dump({k: v.tolist() for k, v in diff_self_dict.items()}, fp)
        with open(deriv_ee_fname, 'w') as fp:
            j = json.dump({'BB': diff_EE_dict['BB'].tolist()}, fp)
        dataLs = np.column_stack((Ls_to_get))
        header0 = "Ls_to_get" 
        output_name0 = "derivs/Ls_to_get.dat"%(dl)
        np.savetxt(output_name0, dataLs, header=header0)
    
    return diff_EE_dict, diff_phi_dict, diff_self_dict

########################################################################################################################

def get_cov(TT, EE, TE, PP, TP, EP):

    C = np.zeros( (3,3) ) #TT, EE, PP
    C[0,0] = TT
    C[1,1] = EE
    C[0,1] = C[1,0] = TE

    C[2,2] = PP
    C[0,2] = C[2,0] = TP
    C[1,2] = C[2,1] = 0. ##EP

    return np.mat( C )

########################################################################################################################

def get_fisher_inv(F_mat):

    F_mat = np.asarray(F_mat)

    Flen = len(F_mat)
    all_inds = np.arange(Flen)

    F_mat_diag = np.diag(F_mat)
    good_inds = np.where(F_mat_diag > 0)[0]
    Flen_refined = len(good_inds)

    #from IPython import embed; embed(); sys.exit()

    F_mat_refined = []
    used_i, used_j = [], []
    for i in all_inds:
        for j in all_inds:
            if i in good_inds and j in good_inds: 
                F_mat_refined.append( (F_mat[j, i]) )
                used_i.append(i)
                used_j.append(j)
    used_i = np.asarray(used_i)
    used_j = np.asarray(used_j)
    F_mat_refined = np.asarray( F_mat_refined ).reshape( (Flen_refined, Flen_refined) )
    C_mat_refined = np.linalg.inv(F_mat_refined)

    C_mat = np.zeros(F_mat.shape)
    C_mat[used_j, used_i] = C_mat_refined.flatten()

    return C_mat

########################################################################################################################

def get_fisher_mat(els, cl_deriv_dict, delta_cl_dict, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_cl_dict.values()[0] ) )

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    for lcntr, l in enumerate( els ):

        TT, EE, TE = 0., 0., 0.
        Tphi = Ephi = PP = 0.
        if 'TT' in delta_cl_dict:
            TT = delta_cl_dict['TT'][lcntr]
        if 'EE' in delta_cl_dict:
            EE = delta_cl_dict['EE'][lcntr]
        if 'TE' in delta_cl_dict:
            TE = delta_cl_dict['TE'][lcntr]
        if with_lensing:
            Tphi, Ephi, PP = delta_cl_dict['Tphi'][lcntr], delta_cl_dict['Ephi'][lcntr], delta_cl_dict['PP'][lcntr]

        ##############################################################################
        #null unused fields
        null_TT, null_EE, null_TE = 0, 0, 0
        if l<min_l_temp or l>max_l_temp:
            null_TT = 1
        if l<min_l_pol or l>max_l_pol: 
            null_EE = 1
            null_TE = 1
        null_PP = 0 #Lensing noise curves already have pretty large noise outside desired L range
        #if l<min_l_TE or l>max_l_TE:  
        #    null_TE = 1

        #20200611
        if 'TT' not in all_pspectra_to_use:
            null_TT = 1
        if 'EE' not in all_pspectra_to_use:
            null_EE = 1
        if 'TE' not in all_pspectra_to_use:
            #if 'TT' not in pspectra_to_use and 'EE' not in pspectra_to_use:
            #    null_TE = 1
            if 'TT' in pspectra_to_use and 'EE' in pspectra_to_use:
                null_TE = 0
            else:
                null_TE = 1
        if ['TT', 'EE', 'TE'] in pspectra_to_use:
            null_TT = 0
            null_EE = 0
            null_TE = 0
        #20200611

        ##if (null_TT and null_EE and null_TE): continue# and null_PP): continue
        if (null_TT and null_EE and null_TE and null_PP): continue
        ##############################################################################
        #get covariance matrix and its inverse
        COV_mat_l = get_cov(TT, EE, TE, PP, Tphi, Ephi)
        inv_COV_mat_l = linalg.pinv2(COV_mat_l)
        #print(COV_mat_l); sys.exit()
        ##############################################################################
        #get the parameter combinations
        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])

        ##############################################################################

        for (p,p2, pcnt, pcnt2) in param_combinations:


            TT_der1, EE_der1, TE_der1 = 0., 0., 0.
            TT_der2, EE_der2, TE_der2 = 0., 0., 0.

            if 'TT' in cl_deriv_dict[p]:
                TT_der1 = cl_deriv_dict[p]['TT'][lcntr]
                TT_der2 = cl_deriv_dict[p2]['TT'][lcntr]
            if 'EE' in cl_deriv_dict[p]:
                EE_der1 = cl_deriv_dict[p]['EE'][lcntr]
                EE_der2 = cl_deriv_dict[p2]['EE'][lcntr]
            if 'TE' in cl_deriv_dict[p]:
                TE_der1 = cl_deriv_dict[p]['TE'][lcntr]
                TE_der2 = cl_deriv_dict[p2]['TE'][lcntr]

            if with_lensing:
                PP_der1, TPhi_der1, EPhi_der1 = cl_deriv_dict[p]['PP'][lcntr], cl_deriv_dict[p]['Tphi'][lcntr], cl_deriv_dict[p]['Ephi'][lcntr]
                PP_der2, TPhi_der2, EPhi_der2 = cl_deriv_dict[p2]['PP'][lcntr], cl_deriv_dict[p2]['Tphi'][lcntr], cl_deriv_dict[p2]['Ephi'][lcntr]
            else:
                PP_der1 = PP_der2 = 0.
                TPhi_der1 = TPhi_der2 = 0. 
                EPhi_der1 = EPhi_der2 = 0.

            if null_TT: TT_der1 = TT_der2 = TPhi_der1 = TPhi_der2 = 0
            if null_EE: EE_der1 = EE_der2 = EPhi_der1 = EPhi_der2 = 0
            if null_TE: TE_der1 = TE_der2 = 0
            if null_PP: PP_der1 = PP_der2 = 0

            fprime1_l_vec = get_cov(TT_der1, EE_der1, TE_der1, PP_der1, TPhi_der1, EPhi_der1)
            fprime2_l_vec = get_cov(TT_der2, EE_der2, TE_der2, PP_der2, TPhi_der2, EPhi_der2)

            curr_val = np.trace( np.dot( np.dot(inv_COV_mat_l, fprime1_l_vec), np.dot(inv_COV_mat_l, fprime2_l_vec) ) )
            if (0):#curr_val>0:
                print(curr_val, p,p2, pcnt, pcnt2, lcntr, l); sys.exit()

            F[pcnt2,pcnt] += curr_val

    return F   
########################################################################################################################

def get_fisher_mat2(els, cl_deriv_dict, delta_cl_dict, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None, delta_cl_dict_nongau = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_cl_dict.values()[0] ) )

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    TT_filter = np.ones(len(els))
    TT_filter[0:3000] = 1

    for lcntr, l in enumerate( els ):

        TT, EE, TE = 0., 0., 0.
        Tphi = Ephi = PP = 0.
        TTTT = delta_cl_dict['TTTT'][lcntr]
        TTEE = delta_cl_dict['TTEE'][lcntr]
        TTTE = delta_cl_dict['TTTE'][lcntr]
        EEEE = delta_cl_dict['EEEE'][lcntr]
        EETE = delta_cl_dict['EETE'][lcntr]
        TETE = delta_cl_dict['TETE'][lcntr]

        ##############################################################################
        #get covariance matrix and its inverse
        
        COV_mat_l = np.zeros((len(delta_cl_dict)))
        COV_mat_l_2d = np.array((delta_cl_dict['TTTT'][lcntr], delta_cl_dict['TTEE'][lcntr], delta_cl_dict['TTEE'][lcntr], delta_cl_dict['EEEE'][lcntr]))
        for i, poweri in enumerate(delta_cl_dict):
            COV_mat_l[i] = delta_cl_dict[poweri][lcntr]
            #clpi = delta_cl_dict[poweri]
        COV_mat_l = np.mat(COV_mat_l.reshape(3,3))
        COV_mat_l_2d = np.mat(COV_mat_l_2d.reshape(2,2))
        inv_COV_mat_l = linalg.pinv2(COV_mat_l)
        inv_COV_mat_l_2d = linalg.pinv2(COV_mat_l_2d)

        ##############################################################################
        #get the parameter combinations
        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])

        ##############################################################################

        for (p,p2, pcnt, pcnt2) in param_combinations:


            TT_der1, EE_der1, TE_der1 = 0., 0., 0.
            TT_der2, EE_der2, TE_der2 = 0., 0., 0.

            if 'TT' in cl_deriv_dict[p]:
                TT_der1 = cl_deriv_dict[p]['TT'][lcntr]*TT_filter[lcntr]
                TT_der2 = cl_deriv_dict[p2]['TT'][lcntr]*TT_filter[lcntr]
            if 'EE' in cl_deriv_dict[p]:
                EE_der1 = cl_deriv_dict[p]['EE'][lcntr]
                EE_der2 = cl_deriv_dict[p2]['EE'][lcntr]
            if 'TE' in cl_deriv_dict[p]:
                TE_der1 = cl_deriv_dict[p]['TE'][lcntr]
                TE_der2 = cl_deriv_dict[p2]['TE'][lcntr]


            if with_lensing:
                PP_der1, TPhi_der1, EPhi_der1 = cl_deriv_dict[p]['PP'][lcntr], cl_deriv_dict[p]['Tphi'][lcntr], cl_deriv_dict[p]['Ephi'][lcntr]
                PP_der2, TPhi_der2, EPhi_der2 = cl_deriv_dict[p2]['PP'][lcntr], cl_deriv_dict[p2]['Tphi'][lcntr], cl_deriv_dict[p2]['Ephi'][lcntr]
            else:
                PP_der1 = PP_der2 = 0.
                TPhi_der1 = TPhi_der2 = 0. 
                EPhi_der1 = EPhi_der2 = 0.


            fprime1_l_vec = np.array((TT_der1, EE_der1, TE_der1))
            fprime2_l_vec = np.array((TT_der2, EE_der2, TE_der2))

            fprime1_l_vec_2d = np.array((TT_der1, EE_der1))
            fprime2_l_vec_2d = np.array((TT_der2, EE_der2))

            curr_val = np.einsum('i,ij,j->', fprime1_l_vec, inv_COV_mat_l, fprime2_l_vec)
            #curr_val = np.einsum('i,ij,j->', fprime1_l_vec_2d, inv_COV_mat_l_2d, fprime2_l_vec_2d)

            F[pcnt2,pcnt] += curr_val

    return F   
########################################################################################################################

########################################################################################################################

def get_fisher_mat3(els, cl_deriv_dict, delta_cl_dict, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None, delta_cl_dict_nongau = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    Ftt = np.zeros([npar,npar])
    Fee = np.zeros([npar,npar])
    Fttee = np.zeros([npar,npar])
    Ftol = np.zeros([npar,npar])
    F_nongau = np.zeros([npar,npar])
    #els = np.arange( len( delta_cl_dict.values()[0] ) )

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    PP_filter = np.zeros(len(els))
    TT_filter = np.zeros(len(els))
    TT_filter[30:5000] = 1
    PP_filter[30:5000] = 1

    clname = ['TT','EE','TE']

    cov_dict = {}
    covmat = [1./delta_cl_dict[i] for i in delta_cl_dict]
    covmat = np.asarray(covmat)
    covnames = [i for i in delta_cl_dict]

    XY_list = [i[0] + i[1] for i in covnames]
    WZ_list = [i[2] + i[3] for i in covnames]
    #covmat = covmat.reshape(3,3,len(els))

    for i in cl_deriv_dict:
        cl_deriv_dict[i]['TT'] = cl_deriv_dict[i]['TT']*TT_filter
        cl_deriv_dict[i]['EE'] = cl_deriv_dict[i]['EE']*PP_filter
        cl_deriv_dict[i]['TE'] = cl_deriv_dict[i]['TE']*PP_filter
        cl_deriv_dict[i]['ET'] = cl_deriv_dict[i]['TE']
 
    '''
    for i in np.arange(len(els)):
        inv_COV_mat_l[:,:,i] = linalg.pinv2(COV_mat_l[:,:,i])
    #inv_COV_mat_l = linalg.pinv2(COV_mat_l)
    '''
    for pcnt,p in enumerate(params):
        for pcnt2,p2 in enumerate(params):
            vect1 = np.asarray([cl_deriv_dict[p][xyi] for xyi in XY_list])
            vect2 = np.asarray([cl_deriv_dict[p2][wzi] for wzi in WZ_list])
            TT_der1 = cl_deriv_dict[p]['TT']
            TT_der2 = cl_deriv_dict[p2]['TT']
            EE_der1 = cl_deriv_dict[p]['EE']
            EE_der2 = cl_deriv_dict[p2]['EE']
            TE_der1 = cl_deriv_dict[p]['TE']
            TE_der2 = cl_deriv_dict[p2]['TE']                
            vect1T = [TT_der1, EE_der1]
            vect2T = [TT_der2, EE_der2]
            #fij = np.einsum('il, il, il->', vect1, covmat, vect2)
            fij1 = np.einsum('l, l, l->', vect1T[0], covmat[0], vect2T[0])
            fij2 = np.einsum('l, l, l->', EE_der1, 1./delta_cl_dict['EEEE'], EE_der2)
            fij3 = np.einsum('l, l, l->', TT_der1, 1./delta_cl_dict['TTEE'], EE_der2)
            Ftt[pcnt, pcnt2] = fij1
            Fee[pcnt, pcnt2] = fij2
            Fttee[pcnt, pcnt2] = fij3*2 + fij1 + fij2

    '''
    for pcnt,p in enumerate(params):
        for pcnt2,p2 in enumerate(params):
            TT_der1 = cl_deriv_dict[p]['TT']*TT_filter
            TT_der2 = cl_deriv_dict[p2]['TT']*TT_filter
            EE_der1 = cl_deriv_dict[p]['EE']
            EE_der2 = cl_deriv_dict[p2]['EE']
            TE_der1 = cl_deriv_dict[p]['TE']
            TE_der2 = cl_deriv_dict[p2]['TE']
            vect1 = np.array((TT_der1, EE_der1, TE_der1))
            vect2 = np.array((TT_der2, EE_der2, TE_der2))

            fij = np.einsum('il, ijl, jl->', vect1, covmat, vect2)
            fij2 = np.einsum('il, ijl, jl->', vect1, covmat2, vect2)
            fgij = np.einsum('il, ijl, jl->', vect1, inv_COV_mat_l, vect2)
            F[pcnt, pcnt2] = fij
            F2[pcnt, pcnt2] = fij2
            F_nongau[pcnt, pcnt2] = fgij
    '''

    if delta_cl_dict_nongau is not None:        
        cov_mat = [np.asmatrix(delta_cl_dict_nongau[i]) for i in delta_cl_dict_nongau]
        cov_mat = [i.I for i in cov_mat]
        cov_mat = np.asarray(cov_mat)
        cov_mat = cov_mat.reshape(3,3,len(els), len(els))
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                TT_der1 = cl_deriv_dict[p]['TT']*TT_filter
                TT_der2 = cl_deriv_dict[p2]['TT']*TT_filter
                EE_der1 = cl_deriv_dict[p]['EE']
                EE_der2 = cl_deriv_dict[p2]['EE']
                TE_der1 = cl_deriv_dict[p]['TE']
                TE_der2 = cl_deriv_dict[p2]['TE']                
                #vect1 = np.array((TT_der1, EE_der1, TE_der1))
                #vect2 = np.array((TT_der2, EE_der2, TE_der2))
                #fij = np.einsum('im, ijmn, jn->', vect1, cov_mat, vect2)
                vect1 = [TT_der1, EE_der1, TE_der1]
                vect2 = [TT_der2, EE_der2, TE_der2]
                vect1T = [TT_der1]
                vect2T = [TT_der2]
                fij = 0
                for i1, di1 in enumerate(vect1T):
                    for i2, di2 in enumerate(vect1T):
                        fij+= np.einsum('m, mn, n->', di1, cov_mat[i1,i2], di2)
                F_nongau[pcnt, pcnt2] = fij

    return Ftt, Fee, Fttee
########################################################################################################################

def get_fisher_mat4(els, cl_deriv_dict, delta_cl_dict, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None, delta_cl_dict_nongau = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    F_nongau = np.zeros([npar,npar])
    #els = np.arange( len( delta_cl_dict.values()[0] ) )

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    PP_filter = np.zeros(len(els))
    TT_filter = np.zeros(len(els))
    TT_filter[30:5000] = 1
    PP_filter[30:5000] = 1

    clname = ['TT','EE','TE','BB']

    covnames = [i for i in delta_cl_dict]
    if 'PPPP' in delta_cl_dict:
        covnames.remove('PPPP')
    covmat = [np.asmatrix(delta_cl_dict[i]).I for i in covnames]
    covmat = np.asarray(covmat)

    XY_list = [i[0] + i[1] for i in covnames]
    WZ_list = [i[2] + i[3] for i in covnames]
    #covmat = covmat.reshape(3,3,len(els))

    for i in cl_deriv_dict:
        cl_deriv_dict[i]['ET'] = cl_deriv_dict[i]['TE']

    for pcnt,p in enumerate(params):
        for pcnt2,p2 in enumerate(params):
            vect1 = np.asarray([cl_deriv_dict[p][xyi] for xyi in XY_list])
            vect2 = np.asarray([cl_deriv_dict[p2][wzi] for wzi in WZ_list])
            fij = np.einsum('im, imn, in->', vect1, covmat, vect2)
            F[pcnt, pcnt2] = fij

    return F
########################################################################################################################

########################################################################################################################

def get_fisher_mat5(els, cl_deriv_dict, delta_cl_dict, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None, delta_cl_dict_nongau = None, binsize = 10, include_B = False, include_lensing = False):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    F_nongau = np.zeros([npar,npar])

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)


    TT_filter = np.zeros(len(els))
    BB_filter = np.zeros((len(els), len(els)))
    TT_filter[30:5000] = 1
    BB_filter[30:5000, 30:5000] = 1

        
    #newend = (len(els) // binsize) *binsize
    newshape = len(els) // binsize
    newl = np.arange(els[0], els[-1]+1, binsize)
    newend = (len(newl)-1) * binsize
    clname = ['TT','EE','TE']
    #clname = ['TT','EE','TE']


    covnames = [i for i in delta_cl_dict]
    if 'PPPP' in delta_cl_dict:
        covnames.remove('PPPP')

    new_deriv_dict = {}
    new_delta_dict = {}
    for p in params:
        pdict = cl_deriv_dict[p]
        new_deriv_dict[p] = {}
        for keyi in clname:
            filpdict = pdict[keyi]*TT_filter
            cut_del = filpdict[:newend]
            new_del = cut_del.reshape((newshape, -1)).mean(axis = 1)
            new_deriv_dict[p][keyi] = new_del



    if len(delta_cl_dict['TTTT'].shape) == 1:
        for keyi in covnames[:9]:
            cut_delta = delta_cl_dict[keyi][:newend]
            new_delta = cut_delta.reshape((newshape, -1)).mean(axis = 1)
            #new_delta = (cut_delta.reshape((newshape, -1)))**2
            #new_delta = (new_delta.mean(axis=1))**0.5
            new_delta_dict[keyi] = new_delta
    else:
        for keyi in covnames[:9]:
            cut_delta = delta_cl_dict[keyi][:newend, :newend]
            new_delta = cut_delta.reshape((newshape, binsize, newshape, binsize)).mean(axis = 3).mean(1)
            #np.fill_diagonal(new_delta, np.diagonal(new_delta)*10)
            new_delta_dict[keyi] = new_delta * binsize
            

    covmat = {}
    if len(new_delta_dict['TTTT'].shape) == 1:
        covmat = [np.diag(new_delta_dict[i]) for i in covnames[:9]]

    else:
        covmat = [new_delta_dict[i] for i in covnames[:9]]

    covmat = np.asarray(covmat)
    #print(covmat.keys())

    if include_B:
        for p in params:
            pdict = cl_deriv_dict[p]
            pdict['BB'] = pdict['BB'] * TT_filter
            cut_del = pdict['BB'][:newend]
            new_del = cut_del.reshape((newshape, -1)).mean(axis = 1)
            new_deriv_dict[p]['BB'] = new_del
         
        clname.append('BB')
        #talclname = clname.append('BB')
        print("clname ", clname)
        for keyi in clname:
            cut_delta = delta_cl_dict['BB'+keyi][:newend, :newend]
            new_delta = cut_delta.reshape((newshape, binsize, newshape, binsize)).mean(axis = 3).mean(1)
            #new_delta = (cut_delta.reshape((newshape, binsize, newshape, binsize)))**2
            #new_delta = (new_delta.mean(axis = 3).mean(axis=1))**0.5
            new_delta_dict['BB'+keyi] = new_delta * binsize
            #covmat.append(new_delta)


    if include_B:
        covmat = covmat.reshape((3,3,len(newl)-1,len(newl)-1))
        covmat = np.block([[covmat[0,0], covmat[0,1], covmat[0,2],new_delta_dict['BBTT'].T],[covmat[1,0], covmat[1,1], covmat[1,2], new_delta_dict['BBEE'].T],[covmat[2,0], covmat[2,1], covmat[2,2], new_delta_dict['BBTE'].T],[new_delta_dict['BBTT'], new_delta_dict['BBEE'], new_delta_dict['BBTE'],new_delta_dict['BBBB']]])
        #covmat = covmat.reshape((4,4,len(newl)-1,len(newl)-1))
        #covmat = np.block([[covmat[0,0], covmat[0,1], covmat[0,2],covmat[0,3]],[covmat[1,0], covmat[1,1], covmat[1,2], covmat[1,3]],[covmat[2,0], covmat[2,1], covmat[2,2], covmat[2,3]],[covmat[3,0], covmat[3,1], covmat[3,2],covmat[3,3]]])
    else:
        print("not  include B")
        covmat = covmat.reshape((3,3,len(newl)-1,len(newl)-1))
        covmat = np.block([[covmat[0,0], covmat[0,1], covmat[0,2]],[covmat[1,0], covmat[1,1], covmat[1,2]],[covmat[2,0], covmat[2,1], covmat[2,2]]])

    if (0):
        np.savetxt('covmat_binsize%s.npy' %(binsize), covmat)
        sys.exit()
    inv_covmat = np.asmatrix(covmat).I

    for pcnt,p in enumerate(params):
        for pcnt2,p2 in enumerate(params):
            vect1 = np.asarray([new_deriv_dict[p][xyi] for xyi in clname])
            vect2 = np.asarray([new_deriv_dict[p2][wzi] for wzi in clname])
            vect1 = vect1.reshape(-1)
            vect2 = vect2.reshape(-1)
            fij = np.einsum('m, mn, n->', vect1, inv_covmat, vect2)
            F[pcnt, pcnt2] = fij*binsize

    return F, covmat, [new_delta_dict, new_deriv_dict]
########################################################################################################################

def get_fisher_mat6(els, cl_deriv_dict, delta_cl_dict, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None, delta_cl_dict_nongau = None, binsize = 10):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    F_nongau = np.zeros([npar,npar])

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    PP_filter = np.zeros(len(els))
    TT_filter = np.zeros(len(els))
    TT_filter[30:5000] = 1
    PP_filter[30:5000] = 1

    newshape = len(els) // binsize
    newl = np.arange(els[0], els[-1]+1, binsize)
    newend = (len(newl)-1) * binsize
    clname = ['TT','EE','TE']
    #clname = ['TT','EE','TE','BB']

    covnames = [i for i in delta_cl_dict]
    if 'PPPP' in delta_cl_dict:
        covnames.remove('PPPP')

    new_deriv_dict = {}
    new_delta_dict = {}
    for p in params:
        pdict = cl_deriv_dict[p]
        new_deriv_dict[p] = {}
        for keyi in clname:
            cut_del = pdict[keyi][:newend]
            new_del = cut_del.reshape((newshape, -1)).mean(axis = 1)
            new_deriv_dict[p][keyi] = new_del
            
    for keyi in delta_cl_dict:
        cut_delta = delta_cl_dict[keyi][:newend]
        new_delta = cut_delta.reshape((newshape, -1)).mean(axis = 1)
        new_delta_dict[keyi] = new_delta
        

    lenloop = len(new_delta_dict['TTTT'])

    for lcntr in np.arange(lenloop):
        TT, EE, TE = 0., 0., 0.
        ##############################################################################
        #get covariance matrix and its inverse
        
        COV_mat_l = np.zeros((len(new_delta_dict)))
        for i, poweri in enumerate(new_delta_dict):
            COV_mat_l[i] = new_delta_dict[poweri][lcntr]

        COV_mat_l = np.mat(COV_mat_l.reshape(len(clname),len(clname)))
        inv_COV_mat_l = linalg.pinv2(COV_mat_l)

        ##############################################################################
        #get the parameter combinations
        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                param_combinations.append([p,p2, pcnt, pcnt2])

        ##############################################################################

        for (p,p2, pcnt, pcnt2) in param_combinations:


            TT_der1, EE_der1, TE_der1 = 0., 0., 0.
            TT_der2, EE_der2, TE_der2 = 0., 0., 0.

            if 'TT' in cl_deriv_dict[p]:
                TT_der1 = new_deriv_dict[p]['TT'][lcntr]*TT_filter[lcntr]
                TT_der2 = new_deriv_dict[p2]['TT'][lcntr]*TT_filter[lcntr]
            if 'EE' in new_deriv_dict[p]:
                EE_der1 = new_deriv_dict[p]['EE'][lcntr]
                EE_der2 = new_deriv_dict[p2]['EE'][lcntr]
            if 'TE' in new_deriv_dict[p]:
                TE_der1 = new_deriv_dict[p]['TE'][lcntr]
                TE_der2 = new_deriv_dict[p2]['TE'][lcntr]
            if 'BB' in new_deriv_dict[p]:
                BB_der1 = new_deriv_dict[p]['BB'][lcntr]
                BB_der2 = new_deriv_dict[p2]['BB'][lcntr]


            fprime1_l_vec = np.array((TT_der1, EE_der1, TE_der1))
            fprime2_l_vec = np.array((TT_der2, EE_der2, TE_der2))
            #fprime1_l_vec = np.array((TT_der1, EE_der1, TE_der1, BB_der1))
            #fprime2_l_vec = np.array((TT_der2, EE_der2, TE_der2, BB_der2))

            curr_val = np.einsum('i,ij,j->', fprime1_l_vec, inv_COV_mat_l, fprime2_l_vec)

            F[pcnt2,pcnt] += curr_val*binsize


    return F

########################################################################################################################

def fix_params(F_mat, param_names, fix_params):

    #remove parameters that must be fixed    
    F_mat_refined = []
    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 in fix_params or p2 in fix_params: continue
            F_mat_refined.append( (F_mat[pcntr2, pcntr1]) )

    totparamsafterfixing = int( np.sqrt( len(F_mat_refined) ) )
    F_mat_refined = np.asarray( F_mat_refined ).reshape( (totparamsafterfixing, totparamsafterfixing) )

    param_names_refined = []
    for p in param_names:
        if p in fix_params: continue
        param_names_refined.append(p)


    return F_mat_refined, param_names_refined

########################################################################################################################

def add_prior(F_mat, param_names, prior_dic):

    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 == p2 and p1 in prior_dic:
                prior_val = prior_dic[p1]
                F_mat[pcntr2, pcntr1] += 1./prior_val**2.

    return F_mat

########################################################################################################################

def get_ellipse_specs(COV, howmanysigma = 1):
    """
    Refer https://arxiv.org/pdf/0906.4123.pdf
    """
    assert COV.shape == (2,2)
    confsigma_dict  = {1:2.3, 2:6.17, 3: 11.8}

    sig_x2, sig_y2 = COV[0,0], COV[1,1]
    sig_xy = COV[0,1]
    
    t1 = (sig_x2 + sig_y2)/2.
    t2 = np.sqrt( (sig_x2 - sig_y2)**2. /4. + sig_xy**2. )
    
    a2 = t1 + t2
    b2 = t1 - t2

    a = np.sqrt(a2)
    b = np.sqrt(b2)

    t1 = 2 * sig_xy
    t2 = sig_x2 - sig_y2
    theta = np.arctan2(t1,t2) / 2.
    
    alpha = np.sqrt(confsigma_dic[howmanysigma])
    
    #return (a*alpha, b*alpha, theta)
    return (a*alpha, b*alpha, theta, alpha*(sig_x2**0.5), alpha*(sig_y2**0.5))

########################################################################################################################

def get_Gaussian(mean, sigma, minx, maxx, delx):

    x = np.arange(minx, maxx, delx)

    #return x, 1./(2*np.pi*sigma)**0.5 * np.exp( -(x - mean)**2. / (2 * sigma**2.)  )
    return x, np.exp( -(x - mean)**2. / (2 * sigma**2.)  )

########################################################################################################################

def get_nl(els, rms_map_T, rms_map_P = None, fwhm = None, Bl = None, elknee_t = -1, alphaknee_t = 0, elknee_p = -1, alphaknee_p = 0):
    """
    compute nl - white noise + beam
    """

    if rms_map_P == None:
        rms_map_P = rms_map_T * 1.414

    if fwhm is not None:
        fwhm_radians = np.radians(fwhm/60.)
        #Bl = np.exp((-fwhm_radians**2.) * els * (els+1) /2.35)
        sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
        sigma2 = sigma ** 2
        Bl_gau = np.exp(els * (els+1) * sigma2)            

    if Bl is None:
        Bl = Bl_gau

    rms_map_T_radians = rms_map_T * np.radians(1/60.)
    rms_map_P_radians = rms_map_P * np.radians(1/60.)

    nl_TT = (rms_map_T_radians)**2. * Bl
    nl_PP = (rms_map_P_radians)**2. * Bl

    if elknee_t != -1.:
        nl_TT = np.copy(nl_TT) * (1. + (elknee_t * 1./els)**alphaknee_t )
    if elknee_p != -1.:
        nl_PP = np.copy(nl_PP) * (1. + (elknee_p * 1./els)**alphaknee_p )

    return Bl, nl_TT, nl_PP

########################################################################################################################



def get_nl_sys(els, A_phi_sys, alpha_phi_sys, els_pivot = 3000, null_lensing_systematic = False):
    factor_phi_deflection = (els * (els+1) )**2./2./np.pi
    nl_mv_sys = A_phi_sys *((els/ els_pivot)**alpha_phi_sys) * factor_phi_deflection**0
    if null_lensing_systematic:
        nl_mv_sys = nl_mv_sys * 0.
    return nl_mv_sys

########################################################################################################################

def fisher_forecast_Aphiphi(els, cl_deriv_dict, delta_cl_dict, params, pspectra_to_use, min_l = 0, max_l = 6000):

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_cl_dict.values()[0] ) )

    pspectra_to_use_full = np.asarray( ['PP'] )

    for lcntr, l in enumerate( els ):

        if l<min_l or l>max_l:
            continue

        PP = delta_cl_dict['PP'][lcntr]
        COV_mat_l = PP**2.
        COV_mat_l = np.mat( COV_mat_l )
        Cinv_l = sc.linalg.pinv2(COV_mat_l) #made sure that COV_mat_l * Cinv_l ~= I
        #print l, p, p2, fprime1_l_vec, fprime2_l_vec, COV_mat_l

        pspec_combinations = []
        for X in pspectra_to_use:
            for Y in pspectra_to_use:
                xind = np.where(pspectra_to_use_full == X)[0][0]
                yind = np.where(pspectra_to_use_full == Y)[0][0]
                if [Y,X, yind, xind] in pspec_combinations: continue
                pspec_combinations.append([X, Y, xind, yind])

        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])

        for (p,p2, pcnt, pcnt2) in param_combinations:
            for (X,Y, xind, yind) in pspec_combinations:

                der1 = np.asarray( [cl_deriv_dict[p]['PP'][lcntr]] )
                der2 = np.asarray( [cl_deriv_dict[p2]['PP'][lcntr]] )

                fprime1_l_vec = np.zeros(len(der1))
                fprime2_l_vec = np.zeros(len(der2))

                fprime1_l_vec[xind] = der1[xind]
                fprime2_l_vec[yind] = der2[yind]

                #if l > 100:
                #    from IPython import embed; embed()

                curr_val = np.dot(fprime1_l_vec, np.dot( Cinv_l, fprime2_l_vec ))

                F[pcnt2,pcnt] += curr_val

    return F    

########################################################################################################################

def get_delensed_from_lensed(nels, els, cl_uns, cl_tots , cphi, n0, nl_TT, nl_PP, dimx = 1024, dimy = 1024, fftd = 1./60/180):
    #input should be cl_dict
    lmax = els[-1]
    lmin = els[0]
    clt_tot = cl_tots[:,0]
    cle_tot = cl_tots[:,1]
    clb_tot = cl_tots[:,2]
    clte_tot = cl_tots[:,3]
    clt_u = cl_uns[:,0]
    cle_u = cl_uns[:,1]
    clb_u = cl_uns[:,2]
    clte_u = cl_uns[:,3]
    print('lenels',len(els))
    print('leneclt',len(clt_tot))
    ct = interpolate.interp1d(els, clt_tot) #observe
    ce = interpolate.interp1d(els, cle_tot) #observe
    cb = interpolate.interp1d(els, clb_tot) #observe
    te = interpolate.interp1d(els, clte_tot) #observe
    nt = interpolate.interp1d(els, nl_TT) #observe
    ne = interpolate.interp1d(els, nl_PP) #observe
    xs = np.fft.fftfreq(dimx, fftd)
    ys = np.fft.fftfreq(dimy, fftd)
    #xs = np.arange(-6000, 6000, fftd)
    #ys = np.arange(-6000, 6000, fftd)
    dluse = xs[1] - xs[0]
    xx, yy = np.meshgrid(xs, ys)
    l1xs = xx.ravel()
    l1ys = yy.ravel()
    l1s = (l1xs**2 + l1ys**2)**0.5
    idl1 = np.nonzero((l1s < lmax)&(l1s > lmin))
    l1xs = l1xs[idl1]
    l1ys = l1ys[idl1]
    l1s = l1s[idl1]
    delen_tp = np.zeros(len(nels))
    delen_ep = np.zeros(len(nels))
    delen_bp = np.zeros(len(nels))
    delen_tep = np.zeros(len(nels))
    #print(i, " ",Li)

    for i, Li in enumerate(nels):
        if Li%500 == 2:
            print('Li ',Li)

        [lx, ly] = [Li, 0]
        [l2xs, l2ys] = [ lx-l1xs, ly-l1ys]
        l2s = (l2xs**2 + l2ys**2)**0.5
        idl = np.nonzero((l2s < lmax)&(l2s > lmin) )
        l1xs = l1xs[idl]
        l1ys = l1ys[idl]
        l1s = l1s[idl]
        l2xs = l2xs[idl]
        l2ys = l2ys[idl]
        l2s = l2s[idl]
        l1s_dot_L = l1xs*lx + l1ys*ly
        l1_cross_L = l1xs*ly - l1ys*lx
        l1_cross_l2 = l1xs*l2ys - l1ys*l2xs
        l2s_dot_L = l2xs*lx + l2ys*ly
        l1v_dot_l2v = l1xs * l2xs + l1ys*l2ys
        l1l2 = l1s * l2s
        l1L = l1s * Li
        idsin = np.nonzero(l1L)        
        cosphi = l1s_dot_L / l1L
        sinphi = l1_cross_L / l1L
        sin2phi = 2 * cosphi * sinphi
        cos2phi = 2 * cosphi**2 - 1
        wllb = l1v_dot_l2v * sin2phi
        wlle = l1v_dot_l2v * cos2phi
        wllt = l1v_dot_l2v
        tn2 = wllt**2 * ct(l1s) * cphi(l2s) * ct(l1s) / (ct(l1s)+nt(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s))
        bn2 = wllb**2 * ce(l1s) * cphi(l2s) * (1- ce(l1s) / (ce(l1s)+ne(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s)))
        en2 = wlle**2 * ce(l1s) * cphi(l2s) * ce(l1s) / (ce(l1s)+ne(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s))
        ten2 = wllt*wlle * te(l1s) * cphi(l2s) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s))
        ts2 = np.sum(tn2)
        es2 = np.sum(en2)
        bs2 = np.sum(bn2)
        tes2 = np.sum(ten2)
        delen_tp[i] = dluse**2 / (2 * np.pi)**2 * ts2
        delen_ep[i] = dluse**2 / (2 * np.pi)**2 * es2
        delen_bp[i] = dluse**2 / (2 * np.pi)**2 * bs2
        delen_tep[i] = dluse**2 / (2 * np.pi)**2 * tes2

    dcl_dict = {}
    dcl_dict['TT'] = delen_tp
    dcl_dict['EE'] = delen_ep
    dcl_dict['BB'] = delen_bp
    dcl_dict['TE'] = delen_tep
    data = np.column_stack((delen_tp, delen_ep, delen_bp, delen_tep))
    return dcl_dict, data


########################################################################################################################

def get_delensed_from_lensed_cvltion(els, cl_uns, cl_tots , cphi, rho2, nl_TT, nl_PP, dimx = 1024, dimy = 1024, fftd = 1./60/180):
    #input should be cl_dict
    lmax = els[-1]
    lmin = els[0]
    clt_tot = cl_tots[:,0]
    cle_tot = cl_tots[:,1]
    clb_tot = cl_tots[:,2]
    clte_tot = cl_tots[:,3]
    clt_u = cl_uns[:,0]
    cle_u = cl_uns[:,1]
    clb_u = cl_uns[:,2]
    clte_u = cl_uns[:,3]
    print('lenels',len(els))
    print('leneclt',len(clt_tot))
    ct = interpolate.interp1d(els, clt_tot) #observe
    ce = interpolate.interp1d(els, cle_tot) #observe
    cb = interpolate.interp1d(els, clb_tot) #observe
    te = interpolate.interp1d(els, clte_tot) #observe
    nt = interpolate.interp1d(els, nl_TT) #observe
    ne = interpolate.interp1d(els, nl_PP) #observe
    xs = np.fft.fftfreq(dimx, fftd)
    ys = np.fft.fftfreq(dimy, fftd)
    #xs = np.arange(-6000, 6000, fftd)
    #ys = np.arange(-6000, 6000, fftd)
    dluse = xs[1] - xs[0]
    xx, yy = np.meshgrid(xs, ys)
    l1xs = xx.ravel()
    l1ys = yy.ravel()
    l1s = (l1xs**2 + l1ys**2)**0.5
    idl1 = np.nonzero((l1s < lmax)&(l1s > lmin)&(l1s>0))
    l1xs = l1xs[idl1]
    l1ys = l1ys[idl1]
    l1s = l1s[idl1]
    delen_tp = np.zeros(len(els))
    delen_ep = np.zeros(len(els))
    delen_bp = np.zeros(len(els))
    delen_tep = np.zeros(len(els))
    #define lxly power
    lx6 = l1xs**6
    lx5y1 = l1xs**5 * l1ys
    lx4y2 = l1xs**4 * l1ys**2
    lx3y3 = l1xs**3 * l1ys**3
    lx2y4 = l1xs**2 * l1ys**4
    lx1y5 = l1xs**1 * l1ys**5
    ly6 = l1ys**6
    fbs1 = ce(l1s) / l1s**4
    fbs2 = ce(l1s)**2 / (ce(l1s)+ne(l1s)) / l1s**4
    gbs1 = cphi(l1s) / l1s**4
    gbs2 = cphi(l1s) / l1s**4 * rho2(l1s)
    fcovb = np.fft.ifft2(matf) * np.fft.ifft2(matg)
    '''
    tn2 = wllt**2 * ct(l1s) * cphi(l2s) * ct(l1s) / (ct(l1s)+nt(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s))
    en2 = wlle**2 * ce(l1s) * cphi(l2s) * ce(l1s) / (ce(l1s)+ne(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s))
    ten2 = wllt*wlle * te(l1s) * cphi(l2s) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s))
    ts2 = np.sum(tn2)
    es2 = np.sum(en2)
    bs2 = np.sum(bn2)
    tes2 = np.sum(ten2)
    delen_tp[i] = dluse**2 / (2 * np.pi)**2 * ts2
    delen_ep[i] = dluse**2 / (2 * np.pi)**2 * es2
    delen_bp[i] = dluse**2 / (2 * np.pi)**2 * bs2
    delen_tep[i] = dluse**2 / (2 * np.pi)**2 * tes2

    dcl_dict = {}
    dcl_dict['TT'] = delen_tp
    dcl_dict['EE'] = delen_ep
    dcl_dict['BB'] = delen_bp
    dcl_dict['TE'] = delen_tep
    data = np.column_stack((delen_tp, delen_ep, delen_bp, delen_tep))
    '''
    return dcl_dict, data


########################################################################################################################

#def get_delensed_from_lensed(els, cl_uns, cl_tots , cphi, n0, nl_TT, nl_PP, dimx = 1024, dimy = 1024, fftd = 10):

#def calculate_n0(cl_uns, cl_tots, els):
def calculate_n0(nels, els, cl_uns, cl_tots, nl_TT, nl_PP, dimx = 1024, dimy = 1024, fftd = 1./60/180):
    xs = np.fft.fftfreq(dimx, fftd)
    ys = np.fft.fftfreq(dimy, fftd)
    #xs = np.arange(-6000, 6000, 10)
    #xs = np.arange(-6000, 6000, 10)
    #ys = np.linspace(-6000, 6000, 2000)
    dluse = xs[1] - xs[0]
    print('fftd',fftd)
    print('dimx',dimx)
    print('dimy',dimy)
    xx, yy = np.meshgrid(xs, ys)
    l1xs = xx.ravel()
    l1ys = yy.ravel()
    l1s = (l1xs**2 + l1ys**2)**0.5
    print('l1s',l1s)
    idl1 = np.nonzero((l1s < els[-1])&(l1s > els[0]))
    l1xs = l1xs[idl1]
    l1ys = l1ys[idl1]
    l1s = l1s[idl1]
    print('l1s',l1s)
    n0_TT = np.zeros(len(nels))
    n0_TE = np.zeros(len(nels))
    n0_TB = np.zeros(len(nels))
    n0_EE = np.zeros(len(nels))
    n0_EB = np.zeros(len(nels))
    n0_BB = np.zeros(len(nels))
    #n0 = {'TT':n0_TT, 'TE':n0_TE, 'TB':n0_TB, 'EE':n0_EE, 'EB':n0_EB, 'BB':n0_BB}
    n0 = {'TT':n0_TT, 'TE':n0_TE, 'TB':n0_TB, 'EE':n0_EE, 'EB':n0_EB}
    clt_tot = cl_tots[:,0]
    cle_tot = cl_tots[:,1]
    clb_tot = cl_tots[:,2]
    clte_tot = cl_tots[:,3]
    clt_u = cl_uns[:,0]
    cle_u = cl_uns[:,1]
    clb_u = cl_uns[:,2]
    clte_u = cl_uns[:,3]
    print('lenels',len(els))
    print('leneclt',len(clt_tot))
    cunlt = interpolate.interp1d(els, clt_u) #unlensed
    cunle = interpolate.interp1d(els, cle_u) #unlensed
    cunlb = interpolate.interp1d(els, clb_u) #unlensed
    cunlte = interpolate.interp1d(els, clte_u) #unlensed
    nt = interpolate.interp1d(els, nl_TT) #unlensed
    ne = interpolate.interp1d(els, nl_PP) #unlensed
    cmbf = cmb_f(cl_tots, cl_uns, els)
    C_BB = interpolate.interp1d(els, clb_tot + nl_PP) #observe
    C_EE = interpolate.interp1d(els, cle_tot + nl_PP) #observe
    C_TT = interpolate.interp1d(els, clt_tot + nl_TT) #observe
    C_TE = interpolate.interp1d(els, clte_tot) #observe
    for i, Li in enumerate(nels):
        if Li%1000 == 2:
            print('Li ',Li)
        [lx, ly] = [Li, 0]
        [l2xs, l2ys] = [ lx-l1xs, ly-l1ys]
        l2s = (l2xs**2 + l2ys**2)**0.5
        #print('Li ',Li)
        idl = np.nonzero( (l2s < els[-1])&(l2s > els[0]) )
        l1xs = l1xs[idl]
        l1ys = l1ys[idl]
        l1s = l1s[idl]
        l2xs = l2xs[idl]
        l2ys = l2ys[idl]
        l2s = l2s[idl]
        #print('l2s',l2s)
        l1s_dot_L = l1xs*lx + l1ys*ly
        l2s_dot_L = l2xs*lx + l2ys*ly
        l1_dot_l2 = l1s * l2s
        l1v_dot_l2v = l1xs * l2xs + l1ys*l2ys
        l1_cross_l2 = l1xs*l2ys - l1ys*l2xs
        idsin = np.nonzero(l1_dot_l2)        
        sinphi = np.zeros(len(l1s))
        cosphi = np.zeros(len(l1s))
        cosphi[idsin] = l1v_dot_l2v[idsin] / l1_dot_l2[idsin]
        sinphi[idsin] = l1_cross_l2[idsin] / l1_dot_l2[idsin]
        sin2phi = 2 * sinphi * cosphi
        cos2phi = 2 * cosphi**2 - 1
        #intgitem = cmbf.f_EB(l1s, l1s_dot_L, l2s, l2s_dot_L, sin2phi)**2 / C_EE(l1s) / C_BB(l2s)
        ce_dot_cb = C_EE(l1s) * C_BB(l2s)
        ce_dot_ce = 2 * C_EE(l1s) * C_EE(l2s)
        #cb_dot_cb = 2 * C_BB(l1s) * C_BB(l2s)
        ct_dot_ct = 2 * C_TT(l1s) * C_TT(l2s)
        ct_dot_cb = C_TT(l1s) * C_BB(l2s)
        ct_dot_ce1 = C_TT(l1s) * C_EE(l2s)
        ct_dot_ce2 = C_TT(l2s) * C_EE(l1s)
        cte_tot_cte = C_TE(l1s)*C_TE(l2s)
        tes =  ct_dot_ce2 * ct_dot_ce1 - ( cte_tot_cte )**2
        ideb = np.nonzero(ce_dot_cb)
        idee = np.nonzero(ce_dot_ce)
        #idbb = np.nonzero(cb_dot_cb)
        idtt = np.nonzero(ct_dot_ct)
        idtb = np.nonzero(ct_dot_cb)
        idte = np.nonzero(tes)
        feb = cmbf.f_EB(l1s, l1s_dot_L, l2s, l2s_dot_L, sin2phi)
        #print(l1s)
        #print(feb)
        #print(cmbf.ce)
        fee = cmbf.f_EE(l1s, l1s_dot_L, l2s, l2s_dot_L, cos2phi)
        ftt = cmbf.f_TT(l1s, l1s_dot_L, l2s, l2s_dot_L)
        fte1 = cmbf.f_TE(l1s, l1s_dot_L, l2s, l2s_dot_L, cos2phi)
        fte2 = cmbf.f_TE(l2s, l2s_dot_L, l1s, l1s_dot_L, cos2phi)
        ftb = cmbf.f_TB(l1s, l1s_dot_L, sin2phi)
        #fbb = cmbf.f_BB(l1s, l1s_dot_L, l2s, l2s_dot_L, cos2phi)
        intgitem_eb = feb[ideb]**2 / ce_dot_cb[ideb]
        intgitem_ee = fee[idee]**2 / ce_dot_ce[idee]
        #intgitem_bb = fbb[idbb]**2 / cb_dot_cb[idbb]
        intgitem_tt = ftt[idtt]**2 / ct_dot_ct[idtt]
        intgitem_tb = ftb[idtb]**2 / ct_dot_cb[idtb]
        intgitem_te = ( ct_dot_ce2[idte]*fte1[idte]**2 - cte_tot_cte[idte]*fte1[idte]*fte2[idte]) / tes[idte]
        #print(C_BB(l2s))
        s0 = sum(intgitem_eb)
        s1 = sum(intgitem_ee)
        #s2 = sum(intgitem_bb)
        s3 = sum(intgitem_tt)
        s4 = sum(intgitem_tb)
        s5 = sum(intgitem_te)
        #print("s0 ",s0, "s1 ", s1, "s3 ",s3,  "s4 ",s4, "s5 ", s5)
        if s0 != 0 and s1 != 0 and s3 != 0 and s4 !=0 and s5 != 0 :
            n0['EB'][i] = (2 * np.pi)**2 / s0 /dluse**2     # dluse is the integral dl
            n0['EE'][i] = (2 * np.pi)**2 / s1 /dluse**2     # dluse is the integral dl
            n0['TT'][i] = (2 * np.pi)**2 / s3 /dluse**2     # dluse is the integral dl
            n0['TB'][i] = (2 * np.pi)**2 / s4 /dluse**2     # dluse is the integral dl
            n0['TE'][i] = (2 * np.pi)**2 / s5 /dluse**2     # dluse is the integral dl
    return n0

########################################################################################################################

def Nadd(l, A=1e-6, lp = 100, nL=1):
    nadd = A * (l/lp)**nL
    return nadd


########################################################################################################################


class cmb_f:
    def __init__(self, totCL, unlensedCL, ls_camb):
        self.ce = totCL[:,1]
        self.ct = totCL[:,0]
        self.cb = totCL[:,2]
        self.ce_un = unlensedCL[:,1] #* 2 * np.pi / ls_camb / (ls_camb + 1)
        self.ct_un = unlensedCL[:,0] #* 2 * np.pi / ls_camb / (ls_camb + 1)
        self.cb_un = unlensedCL[:,2] #* 2 * np.pi / ls_camb / (ls_camb + 1)
        self.te = totCL[:,3]
        self.te_un = unlensedCL[:,3]
        self.cmb_l = ls_camb
        self.ctfun = interpolate.interp1d(self.cmb_l, self.ct_un)
        self.tefun = interpolate.interp1d(self.cmb_l, self.te_un)
        self.cefun = interpolate.interp1d(self.cmb_l, self.ce_un)
        self.cbfun = interpolate.interp1d(self.cmb_l, self.cb_un)
                

    def f_TT(self,l1, l1_dot_L, l2, l2_dot_L):
        ftt =  self.ctfun(l1)*l1_dot_L + self.ctfun(l2)*l2_dot_L 
        return ftt

    def f_TE(self, l1, l1_dot_L, l2, l2_dot_L, cos2phi):
        fte = self.tefun(l1)*l1_dot_L*cos2phi + self.tefun(l2)*l2_dot_L 
        return fte

    def f_TB(self, l1, l1_dot_L, sin2phi):
        ftb = self.tefun(l1)*l1_dot_L*sin2phi
        return ftb

    def f_EE(self, l1, l1_dot_L, l2, l2_dot_L, cos2phi):
        fee = ( self.cefun(l1)*l1_dot_L + self.cefun(l2)*l2_dot_L ) * cos2phi
        return fee

    def f_EB(self, l1, l1_dot_L, l2, l2_dot_L, sin2phi):
        feb = ( self.cefun(l1)*l1_dot_L + self.cbfun(l2)*l2_dot_L ) * sin2phi
        #print(feb)
        return feb

    def f_BB(self,l1, l1_dot_L, l2, l2_dot_L, cos2phi):
        fbb = ( self.cbfun(l1)*l1_dot_L + self.cbfun(l2)*l2_dot_L ) * cos2phi
        return fbb

