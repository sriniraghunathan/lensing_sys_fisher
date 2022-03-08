import numpy as np, glob
from pylab import *

#fname = 'lensing_noise_curves.npy'
fname = 'lensing_noise_curves_with_pol.npy'

opdic = np.load(fname, allow_pickle = True, encoding='latin1').item()
pl_dic = {}
for expname in sorted( opdic ):
    print(expname)
    pl_dic[expname] = [opdic[expname]['els'], opdic[expname]['cl_kk'], opdic[expname]['Nl_TT'], opdic[expname]['Nl_MV'], opdic[expname]['Nl_MVpol']]

#plot
colorarr = ['orangered', 'limegreen', 'purple', 'olive', 'dodgerblue', 'teal']
ax = subplot(111, yscale = 'log')
for fcnt, expname in enumerate( sorted(pl_dic) ):
    els, cl_kk, nl_tt, nl_mv, nl_mvpol = pl_dic[expname]
    if fcnt == 0: plot(els, cl_kk, color = 'k', label = r'Lensing')
    #plot(els, nl_tt, color = colorarr[fcnt], ls = '--', label = r'%s' %(expname.upper()))
    plot(els, nl_mv, color = colorarr[fcnt], label = r'%s' %(expname.upper()))
xlabel(r'$L$')
ylabel(r'$[L(L+1)]^2 C_L^{\phi\phi} / 2\pi$')
legend(loc = 2, fancybox = 1)
lmax = 5000
xlim(min(els), lmax)
title(r'Lensing noise curves')
show()

    