from core import WhiteCovMatrix
import numpy as np

# Getting a covariance matrix from manually given values
params = dict()
params['efac_20cm'] = 1.1
params['equad_20cm'] = 1e-6
params['efac_10cm'] = 1.05
params['equad_10cm'] = 1e-7
sigma_prefit=np.repeat(1e-7,3)
bcknd_per_toa = ['20cm','10cm','10cm']
aaa = WhiteCovMatrix(params=params,sigma_prefit=sigma_prefit,bcknd_per_toa=bcknd_per_toa)
aaa.get_covmatrix()
print(aaa.get_covmatrix())

# Getting a covariance matrix from simulated values
bbb = WhiteCovMatrix()
config = dict()
config['sigma_prefit'] = 1e-7
config['frac'] = [0.3,0.3,0.4]
config['ntoa'] = 100
config['efac'] = [1.1, 1.2, 1.3]
config['equad'] = [1e-6, 1e-7, 1e-8]
bbb.simulate(config)
print(bbb.get_covmatrix())

# Getting a red noise covariance matrix, using parameters for J1545-4550
params_red = dict()
params_red['alpha'] = 3
params_red['fc'] = 0.1/spc.year
params_red['p0'] = 1.66268e-26
toa = [1000000, 1000010, 1000020]
ccc = SpinCovMatrix(params=params,toa=toa)
