import numpy as np, sys, scipy as sc, os
#from pylab import *
from scipy import linalg
import copy
from scipy import interpolate 
from scipy.interpolate import interp1d

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
def get_cmb_spectra_using_camb(param_dict, which_spectra, step_size_dict_for_derivatives = None, raw_cl = 1, high_low = 0, verbose = True):

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
    pars.InitPower.set_params(ns=param_dict_to_use['ns'], r=param_dict_to_use['r'], As = param_dict_to_use['As'])
    ########################

    ########################
    #get results
    results = camb.get_results(pars)
    ########################

    ########################
    #get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = raw_cl)#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
    ########################

    ########################
    #get only the required ell range since powerspectra start from ell=0 by default
    for keyname in powers:
        powers[keyname] = powers[keyname][param_dict['min_l_limit']:, :]
    els = np.arange(param_dict['min_l_limit'], param_dict['max_l_limit']+1)
    ########################



    ########################
    #add delensedCL
    if which_spectra == 'delensed_scalar':
        if param_dict['uK']:
            powers['total'] *= 1e12
            powers['unlensed_total'] *= 1e12
        cl_phiphi, cl_Tphi, cl_Ephi = powers['lens_potential'].T
        cphifun = interpolate.interp1d(els, cl_phiphi)
        totCL=powers['total']
        unlensedCL=powers['unlensed_total']
        print('lenels ',len(els))
        print('lenecl ',unlensedCL.shape)

        bl, nlT, nlP = get_nl(els, 2., None, 1.)

        #n0s = np.loadtxt('params/generate_n0s.dat') # use this n0 as a test
        nels = np.arange(els[0], els[-1]+100, 100)
        n0s = calculate_n0(nels, els, unlensedCL, totCL, nlT, nlP, dimx = 1024, dimy = 1024, fftd = 1./60/180)
        mv = 1./(1./n0s['EB']+1./n0s['EE']+1./n0s['TT']+1./n0s['TB']+1./n0s['TE'])
        data = np.column_stack((nels,n0s['EB'],n0s['EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv))
        header = "els,n0s['EB'],n0s[q'EE'],n0s['TT'],n0s['TB'],n0s['TE'], mv" 
        output_name = "params/generate_n0s.dat"
        np.savetxt(output_name, data, header=header)
        #rhosq = cphifun(els) / (cphifun(els) + n0fun(els) + Nadd(els))
        #rhosqfun = interpolate.interp1d(els, rhosq)

        n0fun = interpolate.interp1d(nels, mv)

        winf = cphifun(els) / (n0fun(els) + cphifun(els))

        clpp = cl_phiphi * winf**2 * (els*(els+1))**2/2/np.pi

        cls = (unlensedCL.T * els*(els+1)/2/np.pi).T
        cls = np.insert(cls, 0, np.zeros((2, 4)), axis = 0)
        clpp = np.insert(clpp, 0, np.zeros(2), axis = 0)
        thyres = camb.correlations.lensed_cls(cls, clpp, lmax = els[-1])
        #thyrese = thyres[:,1]
        #thyrest = thyres[:,0]
        #thyresb = thyres[:,2]
        #thyreste = thyres[:,3]

        powers[which_spectra] = thyres[param_dict['min_l_limit']:, :]
        powers[which_spectra] = powers[which_spectra] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
        powers[which_spectra] *= 1e-12

        #'''
        delensed_dict,  delensedCL=get_delensed_from_lensed(nels, els, unlensedCL, totCL , cphifun, n0fun,  nlT, nlP, dimx = 1024, dimy = 1024, fftd = 1./60/180)
        #delensed_dict,  delensedCL=get_delensed_from_lensed_cvltion(els, unlensedCL, totCL , cphifun, rhosqfun,  nlT, nlP, dimx = 1024, dimy = 1024, fftd = 1./60/180)
        fundelt = interpolate.interp1d(nels, delensed_dict['TT'])
        fundele = interpolate.interp1d(nels, delensed_dict['EE'])
        fundelb = interpolate.interp1d(nels, delensed_dict['BB'])
        fundelte = interpolate.interp1d(nels, delensed_dict['TE'])
        data = np.column_stack((fundelt(els), fundele(els), fundelb(els), fundelte(els) ))
        powers[which_spectra] = data
        powers[which_spectra] *= 1e-12
        #'''
    ########################



    ########################
    #do we need cl or dl
    if not raw_cl: #20200529: also valid for lensing (see https://camb.readthedocs.io/en/latest/_modules/camb/results.html#CAMBdata.get_lens_potential_cls)
        powers[which_spectra] = powers[which_spectra] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))
    ########################

    ########################    
    #Tcmb factor
    if pars.OutputNormalization == 1:
        powers[which_spectra] = param_dict['T_cmb']**2. *  powers[which_spectra]
    ########################

    ########################
    #K or uK
    if param_dict['uK']:
        powers[which_spectra] *= 1e12
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

    cl_phiphi = cl_phiphi * (els * (els+1))**2. /(2. * np.pi)
    cl_Tphi = cl_Tphi * (els * (els+1))**1.5 /(2. * np.pi)
    cl_Ephi = cl_Ephi * (els * (els+1))**1.5 /(2. * np.pi)
    
    cl_dict['PP'] = cl_phiphi
    cl_dict['Tphi'] = cl_Tphi
    cl_dict['Ephi'] = cl_Ephi

    return pars, els, cl_dict

########################################################################################################################


########################################################################################################################

def get_derivatives(param_dict, which_spectra, step_size_dict_for_derivatives = None, params_to_constrain = None):

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
        dummypars, els, cl_mod_dic_1 = get_cmb_spectra_using_camb(param_dict, which_spectra, step_size_dict_for_derivatives = tmpdic, high_low = 0)
        #compute power with fid-step
        dummypars, els, cl_mod_dic_2 = get_cmb_spectra_using_camb(param_dict, which_spectra, step_size_dict_for_derivatives = tmpdic, high_low = 1)

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
    'As' : 0.1e-9,
    'ns' : 0.010,
    ###'ws' : -1e-2,
    'neff': 0.080,
    'mnu': 0.02,
    ###'YHe': 0.005, 
    ###'Alens': 1e-2, 
    ###'Aphiphi': 1e-2, 
    'thetastar': 0.000050, 
    }
    ref_step_size_dict_for_derivatives = {}
    for p in step_size_dict_for_derivatives:
        if p not in params_to_constrain: continue
        ref_step_size_dict_for_derivatives[p] = step_size_dict_for_derivatives[p]
    return ref_step_size_dict_for_derivatives

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
        delta_cl_dict[XX] = np.sqrt(2./ (2.*els + 1.) / fsky ) * (cl + nl)

    return delta_cl_dict

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

            F[pcnt2,pcnt] += curr_val

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
        tn2 = wllt**2 * ct(l1s) * cphi(l2s) * (1-ct(l1s) / (ct(l1s)+nt(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s)))
        bn2 = wllb**2 * ce(l1s) * cphi(l2s) * (1- ce(l1s) / (ce(l1s)+ne(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s)))
        en2 = wlle**2 * ce(l1s) * cphi(l2s) * (1-ce(l1s) / (ce(l1s)+ne(l1s)) * cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s)))
        ten2 = wllt*wlle * te(l1s) * cphi(l2s) * (1-cphi(l2s) / (cphi(l2s)+n0(l2s)+Nadd(l2s)))
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
    C_BB = interpolate.interp1d(els, clt_tot + nl_TT) #observe
    C_EE = interpolate.interp1d(els, cle_tot + nl_PP) #observe
    C_TT = interpolate.interp1d(els, clb_tot + nl_PP) #observe
    C_TE = interpolate.interp1d(els, clt_tot) #observe
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

