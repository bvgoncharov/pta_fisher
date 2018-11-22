from core import WhiteCovMatrix

params = dict()
params['efac'] = [1,2]
params['equad']=[1e-6,1e-5]
aaa = WhiteCovMatrix(params=params,sigma_prefit=[10,10])
aaa.get_covmatrix()
print(aaa.get_covmatrix())
