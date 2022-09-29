import numpy as np, os, sys

which_spectra_arr = ['delensed_scalar', 'unlensed_total', 'total']
noise_level_arr = np.arange(1., 11., 1.)
make_batch_scripts = False

for which_spectra in which_spectra_arr:
    for noise_level in noise_level_arr:
        opline = 'python3 fisher_diff_noise.py -which_spectra %s -rms_map_T %s' %(which_spectra, noise_level)
        if make_batch_scripts:
            template_fname='batch_jobs/template_slurm.sh'
            opfname = 'batch_jobs/fisher_%s_noise%s.sh' %(which_spectra, noise_level)
            opf=open(opfname, 'w')
            template=open(template_fname, 'r')
            for line in template:
                opf.writelines('%s\n' %(line.strip()))
            opf.writelines('%s\n' %(opline))
            opf.close()
            cmd='sbatch %s' %(opfname)
            os.system(cmd)
            print('\n%s'%(cmd))
        else:
            print(opline)
sys.exit()
