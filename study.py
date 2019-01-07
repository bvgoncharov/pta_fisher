from core import WhiteCovMatrix, SpinCovMatrix, TotalCovMatrix
import numpy as np
import scipy.constants as spc
import pandas as pd
import ipdb

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Getting a covariance matrix from manually given values
params = dict()
params['efac_20cm'] = 1.1
params['equad_20cm'] = 1e-6
params['efac_10cm'] = 1.05
params['equad_10cm'] = 1e-7
sigma_prefit=np.repeat(1e-7,3)
bcknd_per_toa = ['20cm','10cm','10cm']

# Getting a red noise covariance matrix, using parameters for J1545-4550
params_red = dict()
params_red['alpha'] = 3
params_red['fc'] = 0.1/spc.year
params_red['p0'] = 1.66268e-26
toa = np.array([1000000, 1000010, 1000020])
ccc = SpinCovMatrix(params=params_red,toa=toa)

# Test for efac
efac_array = np.linspace(1,2,num=20)
fisher_efacfc = []
error_efacfc = []

for efac_new in efac_array:
  params['efac_10cm'] = efac_new
  aaa = WhiteCovMatrix(params=params,sigma_prefit=sigma_prefit,bcknd_per_toa=bcknd_per_toa)
  list_cov = []
  list_cov.append(aaa) # White noise 3x3 matrix
  list_cov.append(ccc) # Red noise 3x3 matrix
  ddd = TotalCovMatrix(list_cov)
  fisher = ddd.get_fisher()
  fisher_efacfc.append( fisher['efac_10cm']['fc'] )
  error = pd.DataFrame(np.linalg.pinv(fisher),fisher.columns,fisher.index)
  error_efacfc.append( error['efac_10cm']['fc'] )

plt.plot(efac_array,fisher_efacfc)
plt.xlabel('efac_10cm')
plt.ylabel('Fisher(efac_10cm, fc)')
plt.savefig('/home/bgonchar/fc_covariance/fisher_efac_fc.png')
plt.close()

plt.plot(efac_array,error_efacfc)
plt.xlabel('efac_10cm')
plt.ylabel('Error(efac_10cm, fc)')
plt.savefig('/home/bgonchar/fc_covariance/error_efac_fc.png')
plt.close()

# Test for equad
params['efac_10cm'] = 1.05
equad_array = np.linspace(1e-4,1e-8,num=20)
fisher_equadfc = []
error_equadfc = []

for equad_new in equad_array:
  params['equad_10cm'] = equad_new
  aaa = WhiteCovMatrix(params=params,sigma_prefit=sigma_prefit,bcknd_per_toa=bcknd_per_toa)
  list_cov = []
  list_cov.append(aaa) # White noise 3x3 matrix
  list_cov.append(ccc) # Red noise 3x3 matrix
  ddd = TotalCovMatrix(list_cov)
  fisher = ddd.get_fisher()
  fisher_equadfc.append( fisher['equad_10cm']['fc'] )
  error = pd.DataFrame(np.linalg.pinv(fisher),fisher.columns,fisher.index)
  error_equadfc.append( error['equad_10cm']['fc'] )

plt.plot(equad_array,fisher_equadfc)
plt.xscale('log')
plt.xlabel('equad_10cm')
plt.ylabel('Fisher(equad_10cm, fc)')
plt.savefig('/home/bgonchar/fc_covariance/fisher_equad_fc.png')
plt.close()

plt.plot(equad_array,error_equadfc)
plt.xscale('log')
plt.xlabel('equad_10cm')
plt.ylabel('Error(equad_10cm, fc)')
plt.savefig('/home/bgonchar/fc_covariance/error_equad_fc.png')
plt.close()

# Test for derivative step (delta):
params['equad_10cm'] = 1e-7
aaa = WhiteCovMatrix(params=params,sigma_prefit=sigma_prefit,bcknd_per_toa=bcknd_per_toa)
list_cov = []
list_cov.append(aaa) # White noise 3x3 matrix
list_cov.append(ccc) # Red noise 3x3 matrix
ddd = TotalCovMatrix(list_cov)

fisher_stability_delta = []
deltas = np.logspace(-10,-1,num=20)
for idx_delta, delta in enumerate(deltas):
  fisher = ddd.get_fisher(delta=delta)
  fisher_stability_delta.append( np.diag(fisher) )
fisher_stability_delta = pd.DataFrame(fisher_stability_delta,columns=fisher.index)

for col in fisher_stability_delta.columns:
  plt.plot(deltas,fisher_stability_delta[col],label=col)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Delta for numerical derivative')
plt.ylabel('Fisher diagonal value')
plt.legend()
plt.savefig('/home/bgonchar/fc_covariance/fisher_stability_deriv.png')
plt.close()
