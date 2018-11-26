from core import WhiteCovMatrix, SpinCovMatrix, TotalCovMatrix
import numpy as np
import scipy.constants as spc
import pandas as pd
import ipdb

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
print('Covariance matrix AAA for white noise, manually added for all ToA:')
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
print('Covariance matrix BBB for white noise, generated for all ToA:')
print(bbb.get_covmatrix())

# Getting a red noise covariance matrix, using parameters for J1545-4550
params_red = dict()
params_red['alpha'] = 3
params_red['fc'] = 0.1/spc.year
params_red['p0'] = 1.66268e-26
toa = np.array([1000000, 1000010, 1000020])
ccc = SpinCovMatrix(params=params_red,toa=toa)
print('Covariance matrix CCC for spin noise, generated for all ToA:')
print(ccc.get_covmatrix())

# Testing derivative calculations:
print('Derivative of covariance matrix with respect to efac_20cm.')
print('It was used only in first observation.')
print( aaa.get_covmatrix_derivative(param_name='efac_20cm') )

print('Derivative of covariance matrix with respect to alpha')
print( ccc.get_covmatrix_derivative(param_name='alpha') )

print('Derivative of covariance matrix with respect to P0')
print( ccc.get_covmatrix_derivative(param_name='p0') )

# Testing a total covariance matrix from red and white noise
list_cov = []
list_cov.append(aaa) # White noise 3x3 matrix
list_cov.append(ccc) # Red noise 3x3 matrix
ddd = TotalCovMatrix(list_cov)
print('Total covariance matrix:')
print( ddd.get_covmatrix() )
print('Total covariance matrix derivative with respect to efac_20cm:')
print( ddd.get_covmatrix_derivative(param_name='efac_20cm') )
print('Total covariance matrix derivative with respect to alpha:')
print( ddd.get_covmatrix_derivative(param_name='alpha') )

# Testing a Fisher matrix calculation:
print('Fisher information matrix from total matrix')
print( ddd.get_fisher() )
print('Fisher information matrix from white matrix')
print( aaa.get_fisher() )
print('Fisher information matrix from red matrix')
print( ccc.get_fisher() )

# Pseudo-inverting the Fisher matrix:
fisher = aaa.get_fisher()
error_matrix = pd.DataFrame(np.linalg.pinv(fisher),fisher.columns,fisher.index)
print('Minimum error matrix for white noise:')
print(error_matrix)
