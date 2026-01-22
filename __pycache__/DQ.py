
import csv
from datetime import datetime
import time

import numpy as np
import scipy.stats as st
import scipy.linalg as sla
import scipy.sparse.linalg as ssl
import xpress as xp

import gpm_sync as dqs
##################################################################################
#### Dual quaternion power iteration
##################################################################################
def _normVector(dqvec):
    """
    Caculate the norm of a vector of dual qauaternions. It is defined thus:
               n
        sqrt{ Sum { conj(dqvec[j]) * dqvec[j] } }
              j=0
    Here, sqrt is the dual number squre root.
    """
    out = np.sum(dqs.conjugate(dqvec.copy()) @ dqvec, axis=0, keepdims=True)
    return dqs._dnsqrt(out)

def _normalizeVector(dqvec):
    """
    Normalize a vector of dual quaternions by dividing it by the dual number formed by
               n
        sqrt{ Sum { conj(dqvec[j]) * dqvec[j] } }
              j=0
    The vector is saved in block matrix format with shape (4*s, 1). Every block is in DQmat format.
    """
    tmp = dqs.bm2mb(dqvec.copy())
    nn = dqs._dninv(_normVector(tmp))
    out = tmp @ nn
    return dqs.mb2bm(out)

def dqpower(mat, max_iter = 20, x0 = None):
    """
    Run the dual quaternion power iteration on the Hermitian matrix of dual quaternions mat.
    mat is has shape (4*s, 4*s). Every 4x4 block is a dual quaternion in DQmat format.
    Output is an estimate of the dual quaternion eigenvector of the largest eigenvalue. It has shape (4*s, 4). Every 4x4 is a dual quaternion in DQmat format.
    Iteration stops after max_iter iterations.
    """
    # Generate initial guess
    if x0 is None:
        shape = (mat.shape[0]//4, 1)
        out = dqs.mb2bm(dqs.dq2dqmat(dqs.rt2dq(*dqs.randomUniformGaussian(shape))))
        out = _normalizeVector(out)
    else:
        out = x0

    for n in range(max_iter):
        out = mat @ out
        out = _normalizeVector(out)
    return out

