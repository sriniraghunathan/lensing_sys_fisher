import numpy as np, sys, tools
from pylab import *
import matplotlib.patches as patches
import warnings, matplotlib.cbook
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


def get_latex_param_str(param, use_H = False):
    params_str_dic= {\
    'norm_YszM': r'${\rm log}(Y_{\ast})$', 'logYstar': r'${\rm log}(Y_{\ast})$', 'alpha_YszM': r'$\alpha_{_{Y}}$',\
    'beta_YszM': r'$\beta_{_{Y}}$', 'gamma_YszM': r'$\gamma_{_{Y}}$', \
    'sigma_logYszM': r'$\sigma_{_{\rm logY}}$', 'alpha_sigma_logYszM': r'$\alpha_{\sigma}$', 'gamma_sigma_logYszM': r'$\gamma_{\sigma}$', \
    'alpha': r'$\eta_{\rm v}$', 'sigma_8': r'$\sigma_{\rm 8}$', \
    'one_minus_hse_bias': r'$1-b_{\rm HSE}$', 'omega_m': r'$\Omega_{\rm m}$', 'thetastar': r'$\theta_{\ast}$',\
    'h0':r'$h$', 'm_nu':r'$\sum m_{\nu}$ [eV]', 'ombh2': r'$\Omega_{b}h^{2}$', 'omch2': r'$\Omega_{c}h^{2}$', 'w0': r'$w_{0}$', 'wa': r'$w_{a}$', \
    'tau': r'$\tau_{\rm re}$', 'As': r'$A_{\rm s}$', 'ns': r'$n_{\rm s}$', 'neff': r'N$_{\rm eff}$', \
    'rho_snr_mlens': r'$\rho_{\rm SNR-lens}$', \
    'slope_vir': r'$A_{\rm v}$', 'intercept_vir': r'$B_{\rm v}$', \
    'r': r'$r$', \
    'A_phi_sys': r'$A_{\phi}^{\rm sys}$', 'alpha_phi_sys': r'$\alpha_{\phi}^{\rm sys}$', \
    }

    if use_H:
        params_str_dic['h0'] = r'$H_{0}$'

    return params_str_dic[param]

#def get_cosmo_param_pl_chars_dict(param_names, fix_params, desired_params_to_plot, use_H, cosmo_param_pl_chars_dict = None):
def get_cosmo_param_pl_chars_dict(param_names, desired_params_to_plot, use_H, cosmo_param_pl_chars_dict = None):
    if cosmo_param_pl_chars_dict is None:
        cosmo_param_pl_chars_dict = {}

    param_names_to_plot = []
    pcntr_for_plotting = 1
    for pp in desired_params_to_plot:
        if pp in param_names:# and pp not in fix_params:
            param_names_to_plot.append(pp)
            if pp not in cosmo_param_pl_chars_dict:
                cosmo_param_pl_chars_dict[pp] = [pcntr_for_plotting, get_latex_param_str(pp, use_H = use_H)]
            pcntr_for_plotting += 1

    return cosmo_param_pl_chars_dict, param_names_to_plot

def get_sbpl_locs_dic(param_names_to_plot, cosmo_param_pl_chars_dict, offset = 0, show_diagonal = True):
    if show_diagonal:
        totparamstoplot = len(param_names_to_plot)
        sbpl_locs_dic = {}
        for p1 in param_names_to_plot:
            for p2 in param_names_to_plot:
                sbpl_locs_dic[(p1,p2)] = cosmo_param_pl_chars_dict[p1][0] + ((cosmo_param_pl_chars_dict[p2][0]-1 + offset) * totparamstoplot)
    else:
        totparamstoplot = len(param_names_to_plot)
        sbpl_locs_dic = {}
        sbpl = 1
        for p1 in param_names_to_plot:
            for p2 in param_names_to_plot:
                if p1 == p2 or (p2,p1) in sbpl_locs_dic: continue
                sbpl_locs_dic[(p1,p2)] = sbpl_locs_dic[(p2,p1)] = sbpl
                sbpl += 1
    return sbpl_locs_dic

def get_ellipse_specs(COV, howmanysigma = 1):
    """
    Refer https://arxiv.org/pdf/0906.4123.pdf
    """
    assert COV.shape == (2,2)
    confsigma_dic = {1:2.3, 2:6.17, 3: 11.8}

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

    area = confsigma_dic[howmanysigma] * 3.14 * a * b
    #print(area)

    return (a*alpha, b*alpha, theta)

def get_Gaussian(mean, sigma, minx, maxx, delx):

    x = np.arange(minx, maxx, delx)

    #return x, 1./(2*np.pi*sigma)**0.5 * np.exp( -(x - mean)**2. / (2 * sigma**2.)  )
    return x, np.exp( -(x - mean)**2. / (2 * sigma**2.)  )

#def make_triangle_plot(exparr, F_dic, param_dict, tr, tc, param_names, desired_params_to_plot, fix_params, color_dic, ls_dic = None, one_or_two_sigma = 1, fsval = 12, use_percent = False, use_H = False, fix_axis_range_to_xxsigma = 4., lwval = 1.5, upper_or_lower_triangle = 'lower', write_titles = True, show_diagonal = True, legloc = 4):
def make_triangle_plot(exparr, F_dic, param_dict, param_names, desired_params_to_plot, color_dic, ls_dic = None, one_or_two_sigma = 1, fsval = 12, use_percent = False, use_H = False, fix_axis_range_to_xxsigma = 3., lwval = 1., upper_or_lower_triangle = 'lower', write_titles = True, show_diagonal = True, legloc = 4):

    #cosmo_param_pl_chars_dict, param_names_to_plot = get_cosmo_param_pl_chars_dict(param_names, fix_params, desired_params_to_plot, use_H)
    cosmo_param_pl_chars_dict, param_names_to_plot = get_cosmo_param_pl_chars_dict(param_names, desired_params_to_plot, use_H)
    totparamstoplot = len(param_names_to_plot)
    diag_matrix = np.arange( totparamstoplot**2 ).reshape((totparamstoplot, totparamstoplot)) + 1
    sbpl_locs_dic = get_sbpl_locs_dic(param_names_to_plot, cosmo_param_pl_chars_dict, show_diagonal = show_diagonal)
    tr = tc = len(desired_params_to_plot)

    widthvalues_for_axis_limits = {}

    for pcntr1, p1 in enumerate( param_names ):
        widthvalues_for_axis_limits[p1] = 0.
        for pcntr2, p2 in enumerate( param_names ):

            if p1 not in desired_params_to_plot or p2 not in desired_params_to_plot: continue
            #if p1 in fix_params or p2 in fix_params: continue
            if not show_diagonal and (p1==p2): continue

            sbpl = sbpl_locs_dic[(p1,p2)]
            if upper_or_lower_triangle == 'lower' and sbpl not in np.tril(diag_matrix): continue
            if upper_or_lower_triangle == 'upper' and sbpl not in np.triu(diag_matrix): continue

            print(sbpl, p1, p2)
            cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]
            x, deltax = param_dict[p1]
            y, deltay = param_dict[p2]

            #deltax, deltay = x/10., y/10. #rough plotting limits
            epsilon_x, epsilon_y = abs(x/10000.), abs(y/10000.) #for Gaussian 1d curve.

            #if p1 == 'As': x*=1e9
            #if p2 == 'As':  y*=1e9

            if use_H:
                if p1 =='h0': 
                    x*=100.
                    deltax*=100.
                if p2 =='h0': 
                    y*=100.
                    deltay*=100.

            if fix_axis_range_to_xxsigma is not None:
                x1, x2 = x - deltax*fix_axis_range_to_xxsigma, x + deltax*fix_axis_range_to_xxsigma
                y1, y2 = y - deltay*fix_axis_range_to_xxsigma, y + deltay*fix_axis_range_to_xxsigma
            else:
                x1, x2 = x - deltax, x + deltax
                y1, y2 = y - deltay, y + deltay

            p1str = cosmo_param_pl_chars_dict[p1][1]
            p2str = cosmo_param_pl_chars_dict[p2][1]

            ax = subplot(tr, tc, sbpl)#, aspect = 'equal')

            if show_diagonal:
                if sbpl<=(tr*(tc-1)):
                    setp(ax.get_xticklabels(), visible=False)
                else:
                    xlabel(p1str, fontsize = fsval+4);

                if ((sbpl-1)%tc == 0) and totparamstoplot>1 and sbpl!= 1:
                    ylabel(p2str, fontsize = fsval+4);
                else:
                    setp(ax.get_yticklabels(), visible=False)
            else: #if not show_diagonal and totparamstoplot == 2:
                xlabel(p1str, fontsize = fsval+4);
                ylabel(p2str, fontsize = fsval+4);
                #setp(ax.get_xticklabels(), visible=True)
                #setp(ax.get_yticklabels(), visible=True)

            #print(p1, p2, sbpl)
            for expcntr, exp in enumerate( exparr ):

                F_mat = F_dic[exp]
                exp_COV = tools.get_fisher_inv(F_mat)
                #print(exp_COV.shape); sys.exit()
                if (0):
                    try:
                        #exp_COV = sc.linalg.pinv(F_mat)
                        exp_COV = np.linalg.inv(F_mat)
                    except:
                        exp_COV = np.zeros((F_mat.shape))
                        non_zero_inds = np.where(F_mat != 0.)
                        tmplen_non_zero_inds = int( ( len(non_zero_inds[0]) )**0.5 )
                        print(p1, p2, np.diag(F_mat))
                        F_mat_mod = F_mat[non_zero_inds].reshape((tmplen_non_zero_inds, tmplen_non_zero_inds))
                        #exp_COV_mod = sc.linalg.pinv(F_mat_mod)                
                        exp_COV_mod = np.linalg.inv(F_mat_mod)
                        exp_COV[non_zero_inds] = exp_COV_mod.flatten()

                #cov_extract = np.asarray( [exp_COV[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
                cov_extract = []
                for ii in cov_inds_to_extract:
                    cov_extract.append(exp_COV[ii])
                cov_extract = np.asarray( cov_extract ).reshape((2,2))
                if (0):#p1 == p2 and p1 == 'alpha':
                    F_ex = []
                    for ii in cov_inds_to_extract:
                        F_ex.append(F_mat[ii])
                    F_ex = np.asarray( F_ex ).reshape((2,2))
                    print(cov_inds_to_extract, cov_extract, F_ex); #sys.exit()

                #if np.sum(cov_extract)<=1e-10: continue
                if np.diag(F_mat)[pcntr1] == 0. or  np.diag(F_mat)[pcntr2] == 0.: continue


                #marginalised_F = sc.linalg.pinv2(cov_extract)
                #cov_extract = sc.linalg.pinv2(marginalised_F)

                colorarr = color_dic[exp]#color_dic[expcntr]
                if ls_dic is not None:
                    lsarr = ls_dic[expcntr]
                    handlelength = 1.5
                else:
                    lsarr = ['-', '-', '-']
                    handlelength = 1.5
                alphaarr = [1., 0.5, 0.25]
                if len(colorarr) == 1:
                    colorarr = np.tile(colorarr[0], len(alphaarr))

                for ss in range(one_or_two_sigma):
                    
                    if p1 == p2:

                        #print(p1, cov_extract[0,0]**0.5)
                        widthval = cov_extract[0,0]**0.5##/2.35
                        #print(widthval, p1); sys.exit()
                        if use_H and p1 =='h0': 
                            widthval*=100.
                            epsilon_x*=100.

                        #print(x, x1, x2, epsilon_x); sys.exit()
                        if p1 == 'thetastar':
                            epsilon_x = epsilon_x / 10.
                        hor, ver = get_Gaussian(x, widthval, x1, x2, epsilon_x)
                        #labval = r'%.4f' %(widthval)
                        if use_percent:
                            labval = r'%.2f\%%' %(100. * abs(widthval/x))
                        else:
                            #labval = r'%.3g' %(widthval)
                            #labval = r'%.4f' %(widthval)
                            labval = r'%.4g' %(widthval)
                        if (0):#totparamstoplot==1:
                            labval = r'%s: %s' %(exp_dic[exp]['label'], labval)
                        plot(hor, ver, color = colorarr[ss], ls = lsarr[ss], lw = lwval, label = labval)                    
                        if legloc == 1:
                            legfsval = fsval-4
                            handlelength = handlelength*0.7
                            handletextpad = 0.5
                        else:
                            legfsval = fsval-3
                            handletextpad = 0.8
                        legend(loc = legloc, framealpha = 1, fontsize = legfsval, edgecolor = 'None', handletextpad=handletextpad, handlelength = handlelength, numpoints = 1, columnspacing = 1, labelspacing = 0.4)#, ncol = 2)#, handlelength = 2.)

                        #print(x1, x2, p1)
                        xlim(x1, x2)
                        ylim(0., 1.)
                        setp(ax.get_yticklabels(), visible=False); tick_params(axis='y',left='off')
                        p1str_tmp = r'$%s$' %(p1str[:p1str.find('[')].replace('$', ''))
                        if write_titles:
                            title(p1str_tmp, fontsize = fsval+4);

                        if p1 in widthvalues_for_axis_limits:
                            widthvalues_for_axis_limits[p1] = max(widthvalues_for_axis_limits[p1], widthval)

                    else:
                        Ep = get_ellipse_specs(cov_extract, howmanysigma = ss + 1)
                        widthval, heightval = Ep[0], Ep[1]
                        #if widthval<=1e-10 or heightval<=1e-10: continue
                        if np.isnan(widthval) or np.isnan(heightval): continue
                        #print(widthval, heightval, p1, p2)
                        '''
                        if use_H:
                            if p1 =='h0': 
                                heightval*=100.
                            if p2 =='h0':
                                widthval*=100.
                        '''
                        ellipse = patches.Ellipse(xy=[x,y], width=2.*widthval, height=2.*heightval, angle=np.degrees(Ep[2]))

                        ax.add_artist(ellipse)
                        ellipse.set_clip_box(ax.bbox)
                        ellipse.set_facecolor('None')#colorarr[ss])
                        #ellipse.set_facecolor(colorarr[ss]); ellipse.set_alpha(0.8)
                        ellipse.set_edgecolor(colorarr[ss])
                        ellipse.set_linewidth(lwval)
                        ellipse.set_linestyle(lsarr[ss])
                        #ellipse.set_alpha(alphaarr[ss])

                        #print(x1, x2, p1, p2)
                        xlim(x1, x2)
                        ylim(y1, y2)

                        axhline(y, lw = 0.1);axvline(x, lw = 0.1)

            if show_diagonal:
                if legloc == 1: #cmb-hd stuff
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            for label in ax.get_xticklabels(): label.set_fontsize(fsval-3.)
            for label in ax.get_yticklabels(): label.set_fontsize(fsval-3.)

            if (0):
                grid(True, which='major', axis = 'x', lw = 0.5, alpha = 0.1)
                grid(True, which='major', axis = 'y', lw = 0.5, alpha = 0.1)

    if not show_diagonal:
        for pcntr, p in enumerate( param_names ):
            for expcntr, exp in enumerate( exparr ):

                F_mat = F_dic[exp]
                exp_COV = get_fisher_inv(F_mat)
                cov_inds_to_extract = [(pcntr, pcntr), (pcntr, pcntr), (pcntr, pcntr), (pcntr, pcntr)]
                cov_extract = []
                for ii in cov_inds_to_extract:
                    cov_extract.append(exp_COV[ii])
                cov_extract = np.asarray( cov_extract ).reshape((2,2))            
                widthval = cov_extract[0,0]**0.5##/2.35
                if use_H and p1 =='h0': 
                        widthval*=100.
                        epsilon_x*=100.
                if p in widthvalues_for_axis_limits:
                    widthvalues_for_axis_limits[p] = max(widthvalues_for_axis_limits[p], widthval)

    #set axis limits now based on widths obtained
    if fix_axis_range_to_xxsigma is not None:
        for pcntr1, p1 in enumerate( param_names ):
            for pcntr2, p2 in enumerate( param_names ):
                if p1 not in desired_params_to_plot or p2 not in desired_params_to_plot: continue
                #if p1 in fix_params or p2 in fix_params: continue
                if (not show_diagonal) and p1 == p2: continue
                sbpl = sbpl_locs_dic[(p1,p2)]                            
                if sbpl not in np.tril(diag_matrix): continue
                deltax, deltay = widthvalues_for_axis_limits[p1], widthvalues_for_axis_limits[p2]
                if deltax == 0. and deltay == 0.: continue
                x, deltax = param_dict[p1]
                y, deltay = param_dict[p2]
                x1, x2 = x - deltax*fix_axis_range_to_xxsigma, x + deltax*fix_axis_range_to_xxsigma
                y1, y2 = y - deltay*fix_axis_range_to_xxsigma, y + deltay*fix_axis_range_to_xxsigma
                ax = subplot(tr, tc, sbpl)#, aspect = 'equal')
                #print(deltax, deltay, p1, p2)
                xlim(x1, x2)
                if p1 != p2:
                    ylim(y1, y2)

    return 


