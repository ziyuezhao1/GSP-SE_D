
import numpy as np
import scipy.stats as st
import scipy.linalg as sla

##################################################################################
#### Conversion: Matrix of blocks <-> Block matrix
##################################################################################
def mb2bm(mb, block_size = 4):
    """
    Convert a matrix of blocks to a block matrix.
    """
    return mb.swapaxes(1, 2).reshape(mb.shape[0]*block_size, mb.shape[1]*block_size)

def bm2mb(bm, block_size = 4):
    """
    Convert a block matrix to a matrix of blocks.
    """
    return bm.reshape(bm.shape[0]//block_size, block_size, bm.shape[1]//block_size, block_size).swapaxes(1, 2)


##################################################################################
#### Erdos-Renyi graph
##################################################################################
def randomERGraph(n, q):
    """
    Generates the adjacency matrix of an Erdos-Renyi graph of size n. The probability of an edge to exist is q.
    Output has shape (n, n).
    """
    A = st.bernoulli.rvs(q, size=(n, n))#,random_state=8749)
    A = np.triu(A, 1)
    A += A.T
    np.fill_diagonal(A, 1)
    return A

def applyERGraph(adjmat, blkmat, block_size = 4):
    """
    Multiply the adjacency matrix of an Erdos-Renyi graph of n vertices by m*n x m*n matrix blkmat, with m=block_size``.
    The (i, j) block of the output is the (i, j) block of blkmat multiplied by adjmat[i, j].
    """
    return np.kron(adjmat, np.ones((block_size, block_size))) * blkmat

##################################################################################
#### Random observation generation
##################################################################################
def _uniformAxis(shape):
    """
    Uniform vector from the 3 sphere.
    """
    out = st.norm.rvs(size=(*shape, 3))
    out /= sla.norm(out, axis=-1, keepdims=True)
    return out

def _uniformAngle(shape):
    """
    Uniform angle from the interval [0, 2*pi].
    """
    return st.uniform.rvs(loc=0, scale=2*np.pi, size=(*shape, 1))

def _gaussianAngle(shape, sigma=1, mean=0):
    """
    Angle sampled from the normal distribution with given mean and standard deviation sigma.
    """
    return st.norm.rvs(loc=mean, scale=sigma, size=(*shape, 1))

def _gaussianTranslation(shape, sigma=1, mean=0):
    """
    Translation with i.i.d. coordinates with normal distribution with given mean and standard deviation sigma.
    """
    return st.norm.rvs(loc=mean, scale=sigma, size=(*shape, 3))

def _gaussianRotation(shape, sigma=1, mean=0):
    """
    Translation with i.i.d. coordinates with normal distribution with given mean and standard deviation sigma.
    """
    return st.norm.rvs(loc=mean, scale=sigma, size=(*shape, 3,3))


def randomGaussian(shape,sigma1,sigma2):
    """
    Sample rotation from the Gaussian distriubtion on the translation and on the rotation.
    Output is r, t.
    r has shape (*shape, 4).
    t has shape (*shape, 3).
    """
    # Rotation
    noise=st.norm.rvs(loc=0, scale=1, size=(*shape, 4,4))#,random_state=43235)
    # Translation
    noise[:,:,3,0:3] =0
    # Here we do an experiemnt for translation noise is 0, we check if the rotation error will go down.
    noise[:,:,0:3,0:3]=sigma1*noise[:,:,0:3,0:3]
    noise[:,:,0:3,3]=sigma2*noise[:,:,0:3,3]
    noise[:,:,3,3] =  0

    return noise

##################################################################################
#### additional functions
##################################################################################

##################################################################################
#### spetralin and translation estimation
##################################################################################
def __rounder(R):
    """
    Round the 3 x 3 matrix R.
    """
    [u, _, vh] = sla.svd(R.squeeze())

    return u @ np.diag((1, 1, sla.det(u @ vh))).astype(R.dtype.type) @ vh

def projection_od(U, n, d):
    U = np.real(U)  # Ensure U is real
    x_0 = U.copy()  # Create a copy of U for output

    # Iterate through each block
    for i in range(n):
        start_row = i * d
        end_row = start_row + d
        block = U[start_row:end_row, :]
        U_b, _, Vt = np.linalg.svd(block)
        x_0[start_row:end_row, :] = U_b @ Vt

    return x_0

def projection_sod(U1, n, d):
    # Ensure U is real
    # Create a copy of U for output
    U=projection_od(U1, n, d)
    x_0 = np.zeros_like(U1)
    # Iterate through each block
    block1=U[0:d,:]
    for i in range(n):
        start_row = i * d
        end_row = start_row + d
        block = U[start_row:end_row, :] @block1.T
        #U_block, _, Vt = np.linalg.svd(block)
        x_0[start_row:end_row, :] = __rounder(block)

    return x_0

def compute_translation_estimates2(U1,d,n, q, T_hat, R_0,L,result):
    """
    Compute the translation estimates based on the provided parameters.

    Parameters:
    - n: Number of poses (int)
    - L: Input matrix for the computation (ndarray)
    - Y_translation: Translation observations (ndarray)
    - x_0: Initial rotation estimates (shape: (n, 3, 3))
    - dqs: Module or object with the method get_solutiont

    Returns:
    - t_estimate_2_copy_f: Reshaped translation estimates (shape: (3n, 1))
    """
    # Ensure U is real
    # Create a copy of U for output
    U=projection_od(U1, n, d)
    x_0 = np.zeros_like(U1)
    # Iterate through each block
    block1=U[0:d,:]
    for i in range(n):
        start_row = i * d
        end_row = start_row + d
        block = U[start_row:end_row, :] @block1.T
        #U_block, _, Vt = np.linalg.svd(block)
        x_0[start_row:end_row, :] = block

    t_estimate_2 = -0.5*(x_0.T @ T_hat @np.linalg.pinv(L)).flatten(order='F')
    t_estimate_2=np.reshape(t_estimate_2,(3*n,1))

    return t_estimate_2
def combine_rotation_and_translation(n, x_0, t_estimate_2_copy_f):
    """
    Combine rotation and translation parts to construct a homogeneous transformation matrix.

    Parameters:
    - n: Number of poses (int)
    - x_0: Initial rotation estimates (shape: (n, 3, 3))
    - Q: Quaternion or rotation data (shape should match)
    - t_estimate_2_copy_f: Translation estimates reshaped to (3n, 1)

    Returns:
    - new_xt: The constructed 4n x 4 estimate matrix
    """
    
    # Combine rotation and translation parts
    xt = np.hstack((x_0, t_estimate_2_copy_f))  # Combine rotations (3) and translation (1) into (n, 3, 4)
    xt = np.reshape(xt, (n, 3, 4))  # Reshape into (n, 3, 4)

    # Construct the 4n x 4 estimate matrix
    new_xt = np.zeros((n, 4, 3))  # Initialize the new matrix with zeros
    new_xt[:, :, 0:3] = xt.transpose((0, 2, 1))  # Transpose xt to fit shape (n, 4, 3)

    # Create a new row for homogeneous coordinates
    new_row = np.array([[0, 0, 0, 1]])  # Homogeneous coordinate row
    new_row = np.expand_dims(new_row, axis=1)  # Expand dimensions for tiling
    new_row = np.tile(new_row, new_xt.shape[0])  # Tile to match the number of poses
    new_row = np.reshape(new_row, (new_xt.shape[0], 4, 1))  # Reshape for concatenation

    # Concatenate the new row to the estimate matrix
    new_xt = np.concatenate((new_xt, new_row), axis=2)  # Concatenate along the last axis (columns)

    # Transpose back to the final shape
    new_xt = new_xt.transpose((0, 2, 1))  # Transpose to (n, 3, 4)
    
    # Final reshape to (4n, 4)
    new_xt = np.reshape(new_xt, (4 * new_xt.shape[0], 4))  # Shape: (4n, 4)

    return new_xt
