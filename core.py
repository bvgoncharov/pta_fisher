import numpy as np
import ipdb

class Covariance_Matrix(object):
    """ Empty covariance matrix class """
    def __init__(self,params=None):
        self.params = params

class WhiteCovMatrix(Covariance_Matrix):
    def __init__(self,params,sigma_prefit,sigma_type='bayesian'):
        Covariance_Matrix.__init__(self,params)
        self.sigma_type = sigma_type
        self.sigma_prefit = sigma_prefit

    def get_covmatrix(self):
        if self.sigma_type == 'bayesian':
            cvmsig = WhiteCovMatrix.sigma_bayesian(self)
        elif self.sigma_type == 'frequentist':
            cvmsig = WhiteCovMatrix.sigma_frequentist(self)
        return np.diag(cvmsig**2)

    def sigma_bayesian(self):
        sb = np.multiply(self.sigma_prefit,self.params['efac'])
        sb = np.power(sb,2) + np.power(self.params['equad'],2)
        return np.power(sb,0.5)

    def sigma_frequentist(self):
        sf = np.power(self.sigma_prefit,2)+np.power(self.params['equad'],2)
        sf = np.multiply(sf, np.power(self.params['efac'],2))
        return np.power(sf,0.5)

