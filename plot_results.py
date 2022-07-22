import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import pandas as pd
import re
import tools

#s = 'asdf=5;iwantthis123jasd'
#result = re.search('asdf=5;(.*)123jasd', s)
#print(result.group(1))
#exec(open('plot_results.py').read())


'''
delen_results=pd.read_csv('results_delens.txt')
len_results=pd.read_csv('results_len.txt')
unlen_results=pd.read_csv('results_un.txt')

delen_array=np.asarray(delen_results['value'])
len_array=np.asarray(len_results['value'])
un_array=np.asarray(unlen_results['value'])
'''

delen_results=pd.read_csv('results_lensys_delensed_scalar_n1_fwhm1.0.txt')
delen_array=np.asarray(delen_results['value'])
param_list = []

for item in delen_results['sigma']:
    parai = re.search('sigma\((.*)\)',item)
    param_list.append(parai.group(1))

rms_map_T_list = np.arange(1,11,1)
len_sigma_array = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_array2 = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_prior = np.zeros((len(rms_map_T_list), len(param_list)))
delen_sigma_prior2 = np.zeros((len(rms_map_T_list), len(param_list)))
unlen_sigma_array = np.zeros((len(rms_map_T_list), len(param_list)))

for i, item in enumerate(rms_map_T_list):
    delen_results=pd.read_csv('results_lensys_delensed_scalar_n%s_fwhm1.0.txt'%(item))
    delen_results_prior=pd.read_csv('results_prior_lensys_delensed_scalar_n%s_fwhm1.0.txt'%(item))
    delen_results_prior2=pd.read_csv('results_prior4_lensys_delensed_scalar_n%s_fwhm1.0.txt'%(item))
    delen_results2=pd.read_csv('results_nolensys_delensed_delensed_scalar_n%s_fwhm1.0.txt'%(item))
    len_results=pd.read_csv('results_lensys_lensed_scalar_n%s_fwhm1.0.txt'%(item))
    unlen_results=pd.read_csv('results_lensys_unlensed_scalar_n%s_fwhm1.0.txt'%(item))

    delen_sigma_array[i] = np.asarray(delen_results['value'])
    delen_sigma_prior[i] = np.asarray(delen_results_prior['value'])
    delen_sigma_prior2[i] = np.asarray(delen_results_prior2['value'])
    delen_sigma_array2[i] = np.insert(np.asarray(delen_results2['value']),0, np.zeros(2), axis = 0)
    len_sigma_array[i] = np.insert(np.asarray(len_results['value']), 0, np.zeros(2), axis = 0)
    unlen_sigma_array[i] = np.insert(np.asarray(unlen_results['value']), 0, np.zeros(2), axis = 0)




n0s = np.loadtxt('params/generate_n0s_rmsT%s_fwhmm%s.dat'%(rms_map_T_list[i], 1.0))
nels = n0s[:,0]
mv = n0s[:,-1]

A_phi_sys_value=1.0e-18
alpha_phi_sys_value=-2.

nsys = tools.get_nl_sys(nels, A_phi_sys_value, alpha_phi_sys_value)



#fig, ax = plt.subplots(3,3,8, figsize = (12,12))
#for i, axi in enumerate(ax.ravel()):
plt.figure(figsize = (12,12))
for i, item in enumerate(param_list):
    axi = plt.subplot(3,3,i+1)
    axi.plot(rms_map_T_list, delen_sigma_array[:,i])
    axi.plot(rms_map_T_list, delen_sigma_prior[:,i])
    axi.plot(rms_map_T_list, delen_sigma_prior2[:,i])
    axi.plot(rms_map_T_list, delen_sigma_array2[:,i])
    axi.plot(rms_map_T_list, len_sigma_array[:,i])
    axi.plot(rms_map_T_list, unlen_sigma_array[:,i])
    axi.legend(['delensed','delensed_prior(A:1e-17, alpha:1)','delensed_prior(A:5e-19, alpha:0.2)','delensed_nonsys','lensed','unlensed'],loc="lower center")
    #axi.legend(['delensed'],loc="lower center")
    axi.set_title(param_list[i])

axi = plt.subplot(3,3,9)
for ni in rms_map_T_list:
    n0s = np.loadtxt('params/generate_n0s_rmsT%s_fwhmm%s.dat'%(ni, 1.0))
    mv = n0s[:,-1]
    axi.loglog(nels, mv)
#axi.legend([+'nsys')
axi.loglog(nels, nsys,  linewidth=1.5)
plt.tight_layout()
plt.savefig('compare_lensys_diff_noise_sigma_n0sys2.png')









'''
x1 = [1,2,3,4]
x2 = [1.1,2.1,3.1,4.1]
x3 = [1.2,2.2,3.2,4.2]
y = [1,1,1,1]
yerr1 = len_array*1e4
yerr2 = delen_array
yerr3 = un_array

fig, ax = plt.subplots()
ax.errorbar(x1, y, yerr1, fmt='o', linewidth=2, capsize = 6)
ax.errorbar(x2, y, yerr2, fmt='o', linewidth=2, capsize = 6)
ax.errorbar(x3, y, yerr3, fmt='o', linewidth=2, capsize = 6)
plt.savefig("resultfig.png")
'''
