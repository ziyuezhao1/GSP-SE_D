
import time
import numpy as np
import algorithms.gpm_sync as dqs

##################################################################################
### ASE for SE(d)
##################################################################################

def get_min_eigenvectors(M, d):
    """
    Input: 
        M: The input matrix (symmetric, typically a Laplacian or covariance matrix).
        d: The number of smallest eigenvectors to return.
    Output: 
        Q[:, idx]: The smallest d eigenvectors (columns of Q).
    """
    # Compute eigenvalues and eigenvectors
    Lambda, Q = np.linalg.eigh(M)  # eigh() for symmetric/Hermitian matrices
    
    # Sort eigenvalues in ascending order and pick the first d indices
    idx = np.argsort(Lambda)[:d]  # Smallest d eigenvalues
    
    return Q[:, idx]
def get_rotation_translation_data_matrix(Y):
    """
    Input: 
        The data matrix Y;
    Separate the rotation and tranlation data matrix, construct each terms in data matrix M;
    Output: 
        Y_rotation_mat: The rotation data matrix;
        Y_translation: The translation data matrix;
        T_hat: The T data matrix in M, please check the paper for more details of T_hat;
        S: The sigma data matrix in paper.
    """
    Y_Blocks=dqs.bm2mb(Y)
    n=Y_Blocks.shape[0]
    
    # The rotation data matrix
    Y_rotation=Y_Blocks[:,:,0:3,0:3]
    Y_rotation_mat=dqs.mb2bm(Y_rotation,3)
    
    #The translation data matrix T^hat
    Y_translation=Y_Blocks[:,:,0:3,-1]
    Y_translation_mat=Y_translation.swapaxes(1, 2).reshape(Y_translation.shape[0]*3, Y_translation.shape[1]*1)
    
    t_average = np.mean(Y_translation_mat, axis=1)
    t_average = np.reshape(t_average,(n*3,1))
    new_arr=np.zeros((Y_rotation.shape[0]*3,n))
    for i in range(n):
        new_arr[i*3,i]=t_average[i*3]
        new_arr[i*3+1,i]=t_average[i*3+1]
        new_arr[i*3+2,i]=t_average[i*3+2]
    T_hat=n*new_arr-Y_translation_mat
   
    #The sigma data matrix in paper
    
    Sigma=np.zeros((n,n,3,3))
 
    for i in range(n):
        for j in range(n):
            Sigma[i,i,:,:]+=np.dot(np.array(Y_translation[i,j,:]).reshape(-1,1), np.array(Y_translation[i,j,:]).reshape(-1,1).T)
    '''
    Y_reshaped = Y_translation[..., np.newaxis]
    Sigma_optimized = np.einsum('ijkl,ijkm->ilkm', Y_reshaped, Y_reshaped)
    Sigma_optimized = np.diagonal(Sigma_optimized, axis1=0, axis2=1).sum(axis=2).T
    '''
    S=dqs.mb2bm(Sigma, 3)
    
    return Y_rotation_mat,Y_translation,T_hat,S


def ASE(Y,L,result,beta,p,q):
    """
    Input: 
        Y: The data matrix of SE(d);
        L: The laplacian matrix
        q: The probability that a measurement will be present.
        p: The probability a measurement will not be corrupted.
    This algorithm is to calculate the smallest d eigenvectors of data matrix M.
    Output: 
        new_xt: The estimation of G^;
        times: The time used for this algrithm
    """
    Y_rotation_mat,Y_translation,T_hat,S=get_rotation_translation_data_matrix(Y)
    d=T_hat.shape[0]//T_hat.shape[1]
    n=Y_rotation_mat.shape[0] // d
    M=2*(np.kron(result,np.eye(d))-Y_rotation_mat)+beta*(S-0.5*T_hat@ np.linalg.pinv(L) @T_hat.T) 
    
    #Compute R^
    times = {}
    eigen_ev_time_ = time.time()

    # Calculate the four leading eigenvectors of Y
    U=get_min_eigenvectors(M, d)
    U1=U[0:3,:]
    U_b, _, Vt = np.linalg.svd(U1)
    P1 = U_b @ Vt
    #Project on SO(d)
    x_0=dqs.projection_sod(U, n, d)
    
    
    times["eigen_ev_time"] = time.time() - eigen_ev_time_
  
    # Compute t^    
    eigen_calculate_translation_time_ = time.time()
    print("here")
   # Calculate the translation estimates   
    t_estimate_2_copy_f=dqs.compute_translation_estimates2(U,d,n, q, T_hat, x_0,L,result)
    
    
    times["eigen_cal_translation_time"] = time.time() - eigen_calculate_translation_time_
       
    # Combine rotation and translation part
    new_xt=dqs.combine_rotation_and_translation(n, x_0, t_estimate_2_copy_f)
    
    return new_xt,P1, times
        