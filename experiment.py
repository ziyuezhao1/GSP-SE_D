
import csv
from datetime import datetime
import numpy as np
import scipy.linalg as sla
import algorithms.gpm_sync as dqs
import algorithms.Spectral as SP
import scipy.stats as st
##################################################################################
#### Experiment
##################################################################################
# Constants
FIELDS = ["n", "sigma_r", "sigma_t", "p", "q", "rep_no", "mean_rotation_error", "mean_translation_error"]

def getnow():
    """
    Print a formated string of date and time now.
    """
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


def openCSVFile(dirpath, test_name, writeheader=True, fields = FIELDS):
    """
    Open the output CSV file.
    Returns the file handler and a DictWriter object.
    """
    fn = dirpath + "/" + test_name + "-" + getnow() + ".csv"
    f = open(fn, 'w', newline='')
    dw = csv.DictWriter(f, delimiter=',', fieldnames=fields)
    if writeheader:
        dw.writeheader()

    return f, dw

def logmsg(msg):
    """
    Write a message as output.
    """
    print(getnow(), " ::: ", msg)

def random_rotation_matrix(d):
    """
    Generate a random rotation matrix in d-dimensional Euclidean space.
    """
    # Generate a random matrix.
    #rng = np.random.default_rng(seed=1234)
    M = np.random.normal(size=(d, d))
    # Perform QR decomposition to obtain a rotation matrix.
    Q, R = np.linalg.qr(M)
    # Ensure that the diagonal elements of R are positive.
    D = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, D)
    # Ensure that the determinant of Q is 1.
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def ExperimentGeneration(n, p, q,d, sigma1,sigma2,ifgaussian=True):
    
    exist_graph = dqs.randomERGraph(n, q)
    row, col = np.nonzero(exist_graph)
    Orig = np.zeros((n,1,d+1,d+1))
    translation_sum=np.zeros((3,))
    
    for i in range(n):
        translation=st.norm.rvs(loc=0, scale=2, size=(3, ))
        Orig[i,0,0:d,0:d] = random_rotation_matrix(d).T
        Orig[i,0,0:d,d]=translation
        Orig[i,0,d,d]=1
        translation_sum+=translation
    for i in range(n):
        Orig[i,0,0:d,d]=Orig[i,0,0:d,d]-translation_sum/n
         
    Clean_mat = np.zeros((n,n,d+1,d+1))
  
    for i in range(n):
        # 
        Ri_T = Orig[i, 0, :d, :d].T  # R⁻¹ = Rᵀ
        ti = Orig[i, 0, :d, d]
        inv_t = -Ri_T @ ti
    
        for j in range(n):
            # inv(Orig[i]) @ Orig[j]
            Rj = Orig[j, 0, :d, :d]
            tj = Orig[j, 0, :d, d]
        
            Clean_mat[i, j, :d, :d] = Ri_T @ Rj
            Clean_mat[i, j, :d, d] = Ri_T @ tj + inv_t
            Clean_mat[i, j, d, :d] = 0
            Clean_mat[i, j, d, d] = 1
    '''
    for i in range(n):
        for j in range(n):
            
            Clean_mat[i,j,:,:]=np.linalg.inv(Orig[i,0,:,:]) @ Orig[j,0,:,:]
    '''
    result = np.sum(exist_graph, axis=1)
    diag_result = np.diag(result)
    #ExistMat = dqs.applyERGraph(exist_graph, Clean_mat.copy(), block_size = d)
    
    if ifgaussian:
        noise = dqs.randomGaussian((n,n),sigma1,sigma2)
        Y1_mat=Clean_mat+noise
        Y11_mat=dqs.mb2bm(Y1_mat)
        Y_mat = dqs.applyERGraph(exist_graph, Y11_mat,block_size=d+1)
        
    else:
        # Generate non-corrupted entries graph
        corr_graph = dqs.randomERGraph(n, p)
        Corr_mat = np.zeros((n,n,d+1,d+1))
        Clean_m=dqs.mb2bm(Clean_mat)
        for i in range(n):
            for j in range(n):
                Corr_mat[i,j,0:d,0:d]=random_rotation_matrix(d)
                Corr_mat[i,j,0:d,d]=Clean_mat[i,j,0:d,d]+1*st.norm.rvs(loc=0, scale=1, size=(3,))
               
                Corr_mat[i,j,d,d]=1
            Corr_mat[i,i,:,:]=np.eye(d+1)
        Corr=dqs.mb2bm(Corr_mat)
        # Apply the graphs
       
        apply_graphs = lambda M, C,D: dqs.applyERGraph(exist_graph, dqs.applyERGraph(corr_graph, M,D) + dqs.applyERGraph(1-corr_graph, C,D),D)
        Y_mat = apply_graphs(Clean_m,Corr,d+1)

    model_out = {'AdjMat':exist_graph,"diag_result":diag_result, 'Y_mat':Y_mat, 'Clean_mat':Clean_mat, 'Orig':Orig}
    return model_out
def Rotation_Alignment(R_est, R_gt):
    n = R_gt.shape[0]
    d = R_gt.shape[2]
    A = np.zeros((d, d))

    for k in range(n):
        U = R_est[k,0,:,:]
        V = R_gt[k,0,:,:].T
        A += V@U

    U1, _, V1 = np.linalg.svd(A)
    det_U1_V1 = np.linalg.det(np.dot(U1, V1))

    identity_matrix = np.eye(d-1)
    zero_column = np.zeros((d-1, 1))
    zero_row = np.zeros((1, d-1))

    R_align = U1 @ np.block([[identity_matrix, zero_column], [zero_row, det_U1_V1]]) @ V1

    return R_align
def calculateError(estimate_mat,P1, original_mat):
    """
    Calculate the rotational and translation estimation error.
    The function first aligns the estimate to the original and then calculates the error.
    estimate, original have shape (n, 1, 4, 4).
    Output are arrays rerr and terr containing the entry-wise rotation and translational errors, respectively.
    """
    # Calculate best aligner
    r_estimate=estimate_mat[...,0:3,0:3]
    
    t_estimate=estimate_mat[...,0:3,3]
    r_original=original_mat[...,0:3,0:3]
    for i in range(r_estimate.shape[0]):
        r_original[i,0,:,:]=r_original[i,0,:,:].T
    
    t_original=original_mat[...,0:3,3]
    
    ba = Rotation_Alignment(r_estimate, r_original)
    # Apply best aligner to estimate
    r_estimate =  r_estimate @ba.T
    #t_estimate=ba.T@t_estimate# @ ba.T


    r_dist=np.ones(r_estimate.shape[0])
    rr=r_estimate - r_original
    for i in range(0,r_estimate.shape[0]):
        r_dist[i] = np.linalg.norm(rr[i,0,:,:], 'fro')
    
    
    # Translational part
    terr=np.ones(t_estimate.shape[0])
    for i in range(0,t_estimate.shape[0]):
        terr[i] = sla.norm( t_estimate[i,0,:] -ba.T @ t_original[i,0,:], ord=2, axis=-1)
    gerr=r_dist+terr

    return np.max(r_dist), np.max(terr),np.max(gerr)

def experiment(n=500,d=3,sigma1 = 0,sigma2=0,beta=1, p = 1, q = 1, ifgaussian = True):
    """
    Run a complete experiment using both the method of Arrigoni and the dual quaternion method, based on the given ground truth.
    gt_dqmat is the ground truth in dqmat, matrix of blocks format.
    sigma_r sigma_t is the standard deviation of the rotational and translational noise, respectively.
    p is the probability that a measurement will be present.
    q is the probability a measurement will not be corrupted.
    dq_max_iter is the number of iterations to perform in the dual quaternion power iteration.
    irls determines whether IRLS is performed.
    Output are arrays rerr and terr containing the entry-wise rotation and translational errors, respectively:
        rerr_dqmat, terr_dqmat, rerr_mat, terr_mat
    When timeit is True, it returns a dictionary containing the running time measurements in seconds of the eigenvalue solvers and rounding steps for both methods.
    """

    # Generate observations
    Model=ExperimentGeneration(n,p,q,d, sigma1,sigma2,ifgaussian=ifgaussian)
    L=Model["diag_result"]-Model['AdjMat']
    
    # Calculate estimate using the method of eigen   
    estimate_eigen_mat,P1, mat_eigen_times = SP.ASE(Model["Y_mat"],L,Model["diag_result"],beta,p,q)
    estimate_mat=np.reshape(estimate_eigen_mat, (n,1, d+1, d+1))
   
    rerr_eigen_mat, terr_eigen_mat,gerr_mat = calculateError(estimate_mat,P1, Model["Orig"])

    return rerr_eigen_mat,terr_eigen_mat, mat_eigen_times,gerr_mat
