import numpy as np
import math
from math import lgamma

class multivariate_t:
    def __init__(self):
        return
    def rvs(self, mu, Sigma, df, n=1):
        '''generate random variables of multivariate t distribution
        Parameters
        ----------
        input:
        mu = mean (d dimensional numpy array)
        Sigma = scale matrix (d x d numpy array)
        df = degrees of freedom
        log = log scale or not
        -------
        rvs : ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
        '''
        p = mu.size
        x = np.random.chisquare(df, n) / df
        z = np.random.multivariate_normal(np.zeros(p), Sigma, (n,))
        return(mu + z / np.sqrt(x)[:, None]) # same output format as random.multivariate_normal

    def pdf(self, x, mu, Sigma, df, log=False):
        '''
        Multivariate t-student density. Returns the density
        of the function at points specified by x.

        input:
            x = parameter (n x d numpy array)
            mu = mean (d dimensional numpy array)
            Sigma = scale matrix (d x d numpy array)
            df = degrees of freedom
            log = log scale or not

        '''
        p = Sigma.shape[0] # Dimensionality
        dec = np.linalg.cholesky(Sigma)
        R_x_m = np.linalg.solve(dec, np.matrix.transpose(np.matrix(x)-mu.transpose()))
        rss = np.power(R_x_m, 2).sum(axis=0)
        logretval = lgamma(1.0*(p + df)/2) - (lgamma(1.0*df/2) + np.sum(np.log(dec.diagonal())) \
           + p/2 * np.log(math.pi * df)) - 0.5 * (df + p) * np.log1p((rss/df) )
        logretval = np.asarray(logretval)[0]
        if log == False:
            return(np.exp(logretval))
        else:
            return(logretval)

multivariate_t = multivariate_t()