from core import WhiteCovMatrix, SpinCovMatrix, TotalCovMatrix
import numpy as np
import scipy.constants as spc
import pandas as pd
import ipdb

# Temporary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from normal_corner import normal_corner

# Getting a covariance matrix from simulated values
#bbb = WhiteCovMatrix()
#config = dict()
#config['sigma_prefit'] = 1e-7
#config['frac'] = [0.3,0.4,0.3]
#config['ntoa'] = 100
#config['efac'] = [1.1, 1.2, 1.3]
#config['equad'] = [1e-4, 1e-5, 1e-6]
#bbb.simulate(config)

bbb = WhiteCovMatrix()
config = dict()
config['sigma_prefit'] = 1e-7
config['frac'] = [1]
config['ntoa'] = 100
config['efac'] = [1.1]
config['equad'] = [1e-4]
bbb.simulate(config)

# Getting a red noise covariance matrix, using parameters for J1545-4550
params_red = dict()
params_red['alpha'] = 3
params_red['fc'] = 0.1/spc.year
params_red['p0'] = 1.66268e-26
toa = np.linspace(1000000,1000000+spc.year*10,100)
eee = SpinCovMatrix(params=params_red,toa=toa)

# Testing a larger covariance matrix from red and white noise
list_cov = []
list_cov.append(bbb)
list_cov.append(eee)
fff = TotalCovMatrix(list_cov)

fisher = fff.get_fisher()
error_matrix = pd.DataFrame(np.linalg.pinv(fisher),columns=fisher.columns,index=fisher.index)

print(np.linalg.matrix_rank(fisher))
ipdb.set_trace()

params_all = {**bbb.params,**eee.params}
mmm = []
[mmm.append(params_all[ppp]) for ppp in params_all]
mmm2 = np.array(mmm)
mmm = pd.DataFrame(data=mmm,index=params_all)

#mvrand = np.random.multivariate_normal(mmm.transpose().values[0],error_matrix,size=1000000)
#mvrand = pd.DataFrame(mvrand,columns=error_matrix.columns)
#import corner
#corner.corner(mvrand)
#plt.savefig('temp.png')
#plt.close()

fig1 = normal_corner.normal_corner(np.linalg.pinv(fisher),mmm2,mmm.index)
plt.savefig('/home/bgonchar/fc_covariance/test/original.png')
plt.close()

ipdb.set_trace()

# Assume our conditional parameter is the last in rows and columns (D component)
# And set it to wrong value
condpr = 'efac_2'
condval = 3
schurA = error_matrix.loc[error_matrix.index != condpr ,error_matrix.columns != condpr]
schurB = error_matrix.loc[error_matrix.index != condpr ,error_matrix.columns == condpr]
schurC = error_matrix.loc[error_matrix.index == condpr ,error_matrix.columns != condpr]
schurD = error_matrix.loc[error_matrix.index == condpr ,error_matrix.columns == condpr]

newcov = schurA - schurB @ schurD**(-1) @ schurC

newmu = mmm.loc[mmm.index != condpr] + schurB @ schurD**(-1) @ (condval - mmm.loc[mmm.index == condpr])


# Now, plot new conditional distribution
#mvrand = np.random.multivariate_normal(newmu.transpose().values[0],newcov,size=1000000)
#mvrand = pd.DataFrame(mvrand,columns=newcov.columns)
#import corner
#corner.corner(mvrand)
#plt.savefig('temp1.png')
#plt.close()

fig2 = normal_corner.normal_corner(error_matrix,mmm,mmm.index)

# Now, set it to the right value
condpr = 'efac_2'
condval = mmm.loc[mmm.index == condpr] # Set the conditional value to self
schurA = error_matrix.loc[error_matrix.index != condpr ,error_matrix.columns != condpr]
schurB = error_matrix.loc[error_matrix.index != condpr ,error_matrix.columns == condpr]
schurC = error_matrix.loc[error_matrix.index == condpr ,error_matrix.columns != condpr]
schurD = error_matrix.loc[error_matrix.index == condpr ,error_matrix.columns == condpr]

newcov = schurA - schurB @ schurD**(-1) @ schurC

newmu = mmm.loc[mmm.index != condpr] + schurB @ schurD**(-1) @ (condval - mmm.loc[mmm.index == condpr])


# Now, plot new conditional distribution
#mvrand = np.random.multivariate_normal(newmu.transpose().values[0],newcov,size=1000000)
#mvrand = pd.DataFrame(mvrand,columns=newcov.columns)
#import corner
#corner.corner(mvrand)
#plt.savefig('temp2.png')
#plt.close()


ipdb.set_trace()
