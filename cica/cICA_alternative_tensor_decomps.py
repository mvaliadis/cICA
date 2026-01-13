## this file contains alternative tensor decomposition methods for cICA

import numpy as np
from numpy import (
    arange, arctan2, argsort, array, concatenate, cos, diag, dot,
    eye, float32, float64, loadtxt, matrix, multiply, newaxis,
    savetxt, sign, sin, sqrt, zeros
)
from numpy.linalg import eig, pinv

import scipy
from scipy.linalg import svd, qr
from sklearn.decomposition import FastICA

from .helper_functions import *
from .SPM import *



# JADE
def jadeR_from_CM(T, n, verbose=False):
    """
    Blind source separation of real signals with JADE, starting from precomputed cumulant matrices (CM).

    Parameters:
        CM -- A 2D NumPy array of shape (m, m * nbcm), where `m` is the number of sources
              and `nbcm` is the number of cumulant matrices.
        verbose -- Print info on progress. Default is False.

    Returns:
        B -- An n x m separating matrix (NumPy array), such that Y = B * X are separated
             sources extracted from the input data.
    """
    # Dimensions
    m = T.shape[0]
    
    myrank = n*(n+1)//2
    nbcm = myrank
    _,eigV = eig2(T.reshape(m**2, m**2))
    CM = np.matrix(eigV[:, :myrank].reshape(m, -1))


    if verbose:
        print(f"jadeR -> Input cumulant matrices: {nbcm} matrices of size {m}x{m}")

    # Joint diagonalization of the cumulant matrices
    V = eye(m, dtype=np.float64)[:n,:]  # Initialize the rotation matrix
    seuil = 1.0e-6  # Threshold for small angles
    encore = True
    sweep = 0
    updates = 0

    if verbose:
        print("jadeR -> Starting joint diagonalization")

    while encore:
        encore = False
        sweep += 1
        if verbose:
            print(f"jadeR -> Sweep #{sweep}")
        upds = 0

        for p in range(m - 1):
            for q in range(p + 1, m):
                # Indices for cumulant matrices
                Ip = arange(p, m * nbcm, m)
                Iq = arange(q, m * nbcm, m)

                # Compute Givens rotation angle
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])

              

                gg = np.dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + sqrt(ton**2 + toff**2))
                gain = (sqrt(ton**2 + toff**2) - ton) / 4.0

                # Apply Givens rotation if the angle is significant
                if abs(theta) > seuil:
                    encore = True
                    upds += 1
                    c = cos(theta)
                    s = sin(theta)
                    G = np.array([[c, -s], [s, c]])

                    # Update rotation matrix
                    pair = np.array([p, q])
                    
                    V[:, pair] = np.dot(V[:, pair], G)

                    # Update cumulant matrices
                    CM[pair, :] = np.dot(G.T, CM[pair, :])
                    CM[:, concatenate([Ip, Iq])] = np.hstack([
                        c * CM[:, Ip] + s * CM[:, Iq],
                        -s * CM[:, Ip] + c * CM[:, Iq]
                    ])

        updates += upds
        if verbose:
            print(f"jadeR -> Completed sweep #{sweep} with {upds} updates")

    if verbose:
        print(f"jadeR -> Total of {updates} Givens rotations")

    # The separating matrix
    B = V.T

    # Sort rows of B to ensure most energetic components appear first
    A = pinv(B)
    keys = np.argsort(np.linalg.norm(A, axis=0))[::-1]
    B = B[keys, :]

    # Fix the signs of the rows of B
    signs = np.sign(B[:, 0] + 1e-10)  # Avoid zero sign
    B = diag(signs) @ B

    return B





# HTD HTD
def recover_pattern_HTDHTD(k4_b,k4_f,k2_b,k2_f,r,l):
    I=k4_b.shape[0]
    matK4b=k4_b.reshape(I**2,I**2)
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    Diag,A=eig2(matK4b)
    # get flattening of k4_f
    matK4f=k4_f.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)

    D, symV = eig2(sym_matK4f)
    eigtol= 1e-8
    rank_k4_f=D.shape[0] - np.searchsorted(D[::-1], eigtol)
    rank_k4_f = min(rank_k4_f, r+l)
    D = D[:rank_k4_f]
    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T


    
    # learn coefficients of a_i in k4_f
    alist=[]
    for apow in A.T[:r]:
        alpha = (apow.reshape(1,-1) @ V).T
        D1alpha = D1 @ alpha
        scalar=(alpha.T @ D1alpha)
        apowmat=apow.reshape(I,I)
        Diag,eigen=eig2(apowmat)
        a=eigen[:,0].flatten()
        a = a/np.sum(a*a)**0.5
        if scalar != 0:
            k4_f-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)
        alist.append(a)
    a_s=np.array(alist).T

    # recover b_i vectors (foreground patterns) 
    Diag,B=eig2(k4_f.reshape(I**2,I**2))
    
    blist=[]
    for bpow in B.T[:l]:
        bpowmat=bpow.reshape(I,I)
        Diag,eigen=eig2(bpowmat)
        b=eigen[:,0].flatten()
        blist.append(b/np.sum(b*b)**0.5)
    blist=np.array(blist).T

    # construct a matrix with columns Vect(b^{\otimes 2})
    vpowlist=[]
    for a in A[:r]:
        vpowlist.append(a)
    for b in B[:l]:
        vpowlist.append(b)

    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in blist.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@ k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= blist[:,contrastind]
    return a_s,b_s_sorted_contrast


# SPM SPM 
def recover_pattern_SPM_SPM(k4_b,k4_f,k2_b,k2_f,r,l):
    I=k4_b.shape[0]
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s,lambdas=subspace_power_method(k4_b.copy(),n=4,d=I,r=r)
    # get flattening of k4_f
    matK4f=k4_f.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    D, symV = eig2(sym_matK4f)
    eigtol= 1e-8
    rank_k4_f=D.shape[0] - np.searchsorted(D[::-1], eigtol)
    rank_k4_f = min(rank_k4_f,r+l)
    D = D[:rank_k4_f]
    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha
        scalar=(alpha.T @ D1alpha)
        if scalar != 0:
            k4_f-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)
       
    # recover b_i vectors (foreground patterns) and prevent repetitive vectors
    b_s,lambdas_prime=subspace_power_method(k4_f,n=4,d=I,r=l)
    # construct a matrix with columns Vect(b^{\otimes 2})
    vpowlist=[]
    for a in a_s.T:
        vpowlist.append(khatri_rao_power(a.reshape(-1, 1), 2).reshape(-1,1))
    for b in b_s.T:
        vpowlist.append(khatri_rao_power(b.reshape(-1, 1), 2).reshape(-1,1))
    
    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in b_s.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@ k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= b_s[:,contrastind]
    return a_s,b_s_sorted_contrast





def recover_pattern_SPM_JADE(k4_b,k4_f,k2_f,k2_b,r=None,l=None):
    I=k4_b.shape[0]
    if r==None:
        # get flattening of k4_b
        matK4b=k4_b.reshape(I**2,I**2)
        # remove repeat entries from matK
        symind, findsym, symindscale = symmetric_indices(I, 2)
        symindscale = np.sqrt(symindscale)
        findsym = findsym.flatten()
        symind = symind[::-1,:].T @ (I ** np.arange(2))
        sym_matK4b = symindscale.reshape(1, -1) * matK4b[symind][:, symind] * symindscale.reshape(-1, 1)
        D_prime,symV_prime=eig2(sym_matK4b)
        eigtol= 1e-12
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), eigtol)
        # r=min(rank_k4_b,I)
        r=rank_k4_b
    
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s,lambdas=subspace_power_method(k4_b,n=4,d=I,r=r)
    
    # get flattening of k4_f,k4_b
    k4_f_copy=k4_f.copy()
    matK4f=k4_f_copy.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    
    D, symV = eig2(sym_matK4f)

    eigtol= 1e-12
    rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)
    rank_k4_f = min(r+l,rank_k4_f)

    D = D[:rank_k4_f]

    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f

    lambda_prime=[]
    
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha

        scalar=(alpha.T @ D1alpha)[0,0]
        if scalar != 0:
            k4_f_copy-=(1. / scalar)* generate_lowrank_tensor(a.reshape(-1,1),4)
        
        
        lambda_prime.append(1./scalar)

        matK4f=k4_f_copy.reshape(I**2,I**2)

        sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
        D, symV = eig2(sym_matK4f)
        eigtol= 1e-12
        rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)

        D = D[:rank_k4_f]
        V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
        D1 = np.diagflat(1. / D).T
    if l==None:
        l=rank_k4_f

    
   
    
    B = np.array(np.linalg.pinv(jadeR_from_CM(k4_f_copy,l))).T
    B = B/np.linalg.norm(B,axis = 0)


    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in B.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= B[:,contrastind]
    return a_s, b_s_sorted_contrast




def jadeR(X,m):
    """
    Blind separation of real signals with JADE.

    jadeR implements JADE, an Independent Component Analysis (ICA) algorithm
    developed by Jean-Francois Cardoso. More information about JADE can be
    found among others in: Cardoso, J. (1999) High-order contrasts for
    independent component analysis. Neural Computation, 11(1): 157-192. Or
    look at the website: http://www.tsi.enst.fr/~cardoso/guidesepsou.html

    More information about ICA can be found among others in Hyvarinen A.,
    Karhunen J., Oja E. (2001). Independent Component Analysis, Wiley. Or at the
    website http://www.cis.hut.fi/aapo/papers/IJCNN99_tutorialweb/

    Translated into NumPy from the original Matlab Version 1.8 (May 2005) by
    Gabriel Beckers, http://gbeckers.nl .

    Parameters:

        X -- an n x T data matrix (n sensors, T samples). Must be a NumPy array
                or matrix.

        m -- number of independent components to extract. Output matrix B will
                have size m x n so that only m sources are extracted. This is done
                by restricting the operation of jadeR to the m first principal
                components. Defaults to None, in which case m == n.

        verbose -- print info on progress. Default is False.

    Returns:

        An m*n matrix B (NumPy matrix type), such that Y = B * X are separated
        sources extracted from the n * T data matrix X. If m is omitted, B is a
        square n * n matrix (as many sources as sensors). The rows of B are
        ordered such that the columns of pinv(B) are in order of decreasing
        norm; this has the effect that the `most energetically significant`
        components appear first in the rows of Y = B * X.

    Quick notes (more at the end of this file):

    o This code is for REAL-valued signals.  A MATLAB implementation of JADE
        for both real and complex signals is also available from
        http://sig.enst.fr/~cardoso/stuff.html

    o This algorithm differs from the first released implementations of
        JADE in that it has been optimized to deal more efficiently
        1) with real signals (as opposed to complex)
        2) with the case when the ICA model does not necessarily hold.

    o There is a practical limit to the number of independent
        components that can be extracted with this implementation.  Note
        that the first step of JADE amounts to a PCA with dimensionality
        reduction from n to m (which defaults to n).  In practice m
        cannot be `very large` (more than 40, 50, 60... depending on
        available memory)

    o See more notes, references and revision history at the end of
        this file and more stuff on the WEB
        http://sig.enst.fr/~cardoso/stuff.html

    o For more info on NumPy translation, see the end of this file.

    o This code is supposed to do a good job!  Please report any
        problem relating to the NumPY code gabriel@gbeckers.nl

    Copyright original Matlab code: Jean-Francois Cardoso <cardoso@sig.enst.fr>
    Copyright Numpy translation: Gabriel Beckers <gabriel@gbeckers.nl>
    """

    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.

    origtype = X.dtype #float64

    X = matrix(X.astype(float64)) #create a matrix from a copy of X created as a float 64 array

    [n,T] = X.shape

    # m = n

    X -= X.mean(1)

    # whitening & projection onto signal subspace
    # -------------------------------------------

    # An eigen basis for the sample covariance matrix
    [D,U] = eig((X * X.T) / float(T))
    # Sort by increasing variances
    k = D.argsort()
    Ds = D[k]

    # The m most significant princip. comp. by decreasing variance
    PCs = arange(n-1, n-m-1, -1)

    #PCA
    # At this stage, B does the PCA on m components
    B = U[:,k[PCs]].T


    # --- Scaling ---------------------------------
    # The scales of the principal components
    scales = np.sqrt(np.asarray(Ds).flatten()[PCs])
    B = diag(1./scales) * B
    #Sphering
    X = B * X

    # We have done the easy part: B is a whitening matrix and X is white.

    del U, D, Ds, k, PCs, scales

    # NOTE: At this stage, X is a PCA analysis in m components of the real
    # data, except that all its entries now have unit variance. Any further
    # rotation of X will preserve the property that X is a vector of
    # uncorrelated components. It remains to find the rotation matrix such
    # that the entries of X are not only uncorrelated but also `as independent
    # as possible". This independence is measured by correlations of order
    # higher than 2. We have defined such a measure of independence which 1)
    # is a reasonable approximation of the mutual information 2) can be
    # optimized by a `fast algorithm" This measure of independence also
    # corresponds to the `diagonality" of a set of cumulant matrices. The code
    # below finds the `missing rotation " as the matrix which best
    # diagonalizes a particular set of cumulant matrices.

    #Estimation of Cumulant Matrices
    #-------------------------------

    # Reshaping of the data, hoping to speed up things a little bit...
    X = X.T #transpose data to (256, 3)
    # Dim. of the space of real symm matrices
    dimsymm = int( (m * (m + 1)) / 2 ) #6
    # number of cumulant matrices
    nbcm = dimsymm; #6
    # Storage for cumulant matrices
    CM = matrix(zeros([m, m*nbcm], dtype = float64))
    R = matrix(eye(m, dtype=float64)) #[[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]
    # Temp for a cum. matrix
    Qij = matrix(zeros([m, m], dtype = float64))
    # Temp
    Xim = zeros(m, dtype=float64)
    # Temp
    Xijm = zeros(m, dtype=float64)

    # I am using a symmetry trick to save storage. I should write a short note
    # one of these days explaining what is going on here.
    # will index the columns of CM where to store the cum. mats.
    Range = arange(m) #[0 1 2]

    for im in range(m):
        Xim = X[:,im]
        Xijm = multiply(Xim, Xim)
        Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * dot(R[:,im], R[:,im].T)
        CM[:,Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = multiply(Xim, X[:,jm])
            Qij = sqrt(2) * multiply(Xijm, X).T * X / float(T) - R[:,im] * R[:,jm].T - R[:,jm] * R[:,im].T
            CM[:,Range] = Qij
            Range = Range + m

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big
    # m x m*nbcm array.


    # Joint diagonalization of the cumulant matrices
    # ==============================================

    V = matrix(eye(m, dtype=float64)) #[[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]

    Diag = zeros(m, dtype=float64) #[0. 0. 0.]
    On = 0.0
    Range = arange(m) #[0 1 2] 
    for im in range(nbcm): #nbcm == 6
        Diag = diag(CM[:,Range])
        On = On + (Diag * Diag).sum(axis = 0)
        Range = Range + m
    Off = (multiply(CM,CM).sum(axis=0)).sum(axis=0) - On
    # A statistically scaled threshold on `small" angles
    seuil = 1.0e-6 / sqrt(T) #6.25e-08
    # sweep number
    encore = True
    sweep = 0
    # Total number of rotations
    updates = 0
    # Number of rotations in a given seep
    upds = 0
    g = zeros([2,nbcm], dtype=float64) #[[ 0.  0.  0.  0.  0.  0.] [ 0.  0.  0.  0.  0.  0.]]
    gg = zeros([2,2], dtype=float64) #[[ 0.  0.]  [ 0.  0.]]
    G = zeros([2,2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    # Joint diagonalization proper

    while encore:
        encore = False
        sweep = sweep + 1
        upds = 0
        Vkeep = V
        
        for p in range(m-1): #m == 3
            for q in range(p+1, m): #p == 1 | range(p+1, m) == [2]
                
                Ip = arange(p, m*nbcm, m) #[ 0  3  6  9 12 15] [ 0  3  6  9 12 15] [ 1  4  7 10 13 16]
                Iq = arange(q, m*nbcm, m) #[ 1  4  7 10 13 16] [ 2  5  8 11 14 17] [ 2  5  8 11 14 17]
                
                #computation of Givens angle
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = dot(g, g.T)
                ton = gg[0,0] - gg[1,1] # -6.54012319852 4.44880758012 -1.96674621935
                toff = gg[0, 1] + gg[1, 0] # -15.629032394 -4.3847687273 6.72969915184
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff)) #-0.491778606993 -0.194537202087 0.463781701868
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0 #5.87059352069 0.449409565866 2.24448683877
                
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s] , [s, c] ]) # DON"T PRINT THIS! IT"LL BREAK THINGS! HELLA LONG
                    pair = array([p, q]) #don't print this either
                    V[:,pair] = V[:,pair] * G
                    CM[pair,:] = G.T * CM[pair,:]
                    CM[:, concatenate([Ip, Iq])] = np.append( c*CM[:,Ip]+s*CM[:,Iq], -s*CM[:,Ip]+c*CM[:,Iq], axis=1)
                    On = On + Gain
                    Off = Off - Gain
        updates = updates + upds #3 6 9 9

    # print ('Number of iter: %d' % updates)


    # A separating matrix
    # -------------------

    B = V.T * B #[[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008]  [ 1.12505903 -2.42824508  0.92226197]]

    # Permute the rows of the separating matrix B to get the most energetic
    # components first. Here the **signals** are normalized to unit variance.
    # Therefore, the sort is according to the norm of the columns of
    # A = pinv(B)

    A = pinv(B) #[[-3.35031851 -2.14563715  0.60277625] [-2.49989794 -1.25230985 -0.0835184 ] [-2.49501641 -0.67979249  0.12907178]]
    keys = array(argsort(multiply(A,A).sum(axis=0)[0]))[0] #[2 1 0]
    B = B[keys,:] #[[ 1.12505903 -2.42824508  0.92226197] [-0.41923305 -0.84589716  1.41050008] [ 0.17242566  0.10485568 -0.7373937 ]]
    B = B[::-1,:] #[[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
    # just a trick to deal with sign == 0
    b = B[:,0] #[[ 0.17242566] [-0.41923305] [ 1.12505903]]
    signs = array(sign(sign(b)+0.1).T)[0] #[1. -1. 1.]
    B = diag(signs) * B #[[ 0.17242566  0.10485568 -0.7373937 ] [ 0.41923305  0.84589716 -1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
    return B



def recover_pattern_JADE_HTD(Y,k4_b,k4_f,k2_f,k2_b,r=None,l=None):
    I=k4_b.shape[0]
    if r==None:
        # get flattening of k4_b
        matK4b=k4_b.reshape(I**2,I**2)
        # remove repeat entries from matK
        symind, findsym, symindscale = symmetric_indices(I, 2)
        symindscale = np.sqrt(symindscale)
        findsym = findsym.flatten()
        symind = symind[::-1,:].T @ (I ** np.arange(2))
        sym_matK4b = symindscale.reshape(1, -1) * matK4b[symind][:, symind] * symindscale.reshape(-1, 1)
        D_prime,symV_prime=eig2(sym_matK4b)
        eigtol= 1e-12
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), eigtol)
        # r=min(rank_k4_b,I)
        r=rank_k4_b
    
    
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s = np.real(np.array(np.linalg.pinv(jadeR(Y.T,r))))
    a_s = a_s/np.linalg.norm(a_s,axis = 0)
    
    # get flattening of k4_f,k4_b
    k4_f_copy=k4_f.copy()
    matK4f=k4_f_copy.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    
    D, symV = eig2(sym_matK4f)

    eigtol= 1e-12
    rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)
    if l==None:
        l=rank_k4_f
    rank_k4_f = min(r+l,rank_k4_f)

    D = D[:rank_k4_f]

    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f

    lambda_prime=[]
    
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha

        scalar=(alpha.T @ D1alpha)[0,0]
        if scalar != 0:
            k4_f_copy-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)
        
            lambda_prime.append(1./scalar)

        matK4f=k4_f_copy.reshape(I**2,I**2)

        sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
        D, symV = eig2(sym_matK4f)
        eigtol= 1e-12
        rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)

        D = D[:rank_k4_f]
        V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
        D1 = np.diagflat(1. / D).T
    
    
   

    Diag,B=eig2(k4_f_copy.reshape(I**2,I**2))
    # D_foreground= Diag
    M=k4_f_copy.reshape(I**2,I**2)
    blist=[]
    for n,bpow in enumerate(B.T[:l]):
        bpowmat=bpow.reshape(I,I)
        Diag_prime,eigen=eig2(bpowmat)
        b=eigen[:,0].flatten()
        blist.append(b/np.sum(b*b)**0.5)
        bpow=Diag_prime[0]*khatri_rao_power(b.reshape(-1, 1), 2)
        M-=Diag[n]*bpowmat.reshape(-1,1)@bpowmat.reshape(1,-1)
    blist=np.array(blist).T
    # second_residule=np.sum(M*M)/(I**4)
    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in blist.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= blist[:,contrastind]
    return a_s, b_s_sorted_contrast




## FOOBI


def kr(*matrices):
    """
    Compute the Khatri-Rao product of input matrices.
    Args:
        *matrices: A sequence of matrices, each of shape (I_k, K), where K is the same for all matrices.
    Returns:
        np.ndarray: The Khatri-Rao product of the input matrices.
    """
    if len(matrices) == 1 and isinstance(matrices[0], list):
        matrices = matrices[0]

    # Check that all matrices have the same number of columns
    K = matrices[0].shape[1]
    if any(mat.shape[1] != K for mat in matrices):
        raise ValueError("All input matrices must have the same number of columns.")

    # Compute the Khatri-Rao product
    result = matrices[-1]
    for mat in reversed(matrices[:-1]):
        result = np.einsum('ik,jk->ijk', mat, result).reshape(-1, K)

    return result

def phi(X, Y, reduced=False):
    """
    Rank-1 detection mapping.
    Args:
        X (np.ndarray): Matrix of size (I, J).
        Y (np.ndarray): Matrix of size (I, J).
        reduced (bool): If True, returns only the unique elements of the mapping.
    Returns:
        np.ndarray: Vector representation of the rank-1 detection mapping.
    """
    if X.shape != Y.shape:
        raise ValueError("phi: X and Y must have the same shape.")

    I, J = X.shape
    X2 = X.reshape(1,1,I,J)
    Y2 = Y.reshape(1,1,I,J)

    P1 = X.reshape(I,J,1,1) * Y2 + X2* Y.reshape(I,J,1,1) 
    P2 = P1.transpose(2,1,0,3)
    D = P1 - P2

    if reduced:
        TJ = np.tril(np.ones((J, J), dtype=bool), -1)
        TI = np.tril(np.ones((I, I), dtype=bool), -1)
        TIJ = np.logical_and(TJ[None, :, None, :], TI[:, None, :, None]).flatten(order = 'F')
        return D.flatten(order = 'F')[TIJ]
    else:
        return D.flatten(order = 'F')

def rank1_mapping(E, sz, gramian=False):
    """
    Constructs a matricization of the rank-1 detection tensor.
    Args:
        E (np.ndarray): Input matrix of size (I*J, R).
        sz (tuple): Dimensions (I, J).
        gramian (bool): If True, computes the Gramian of the rank-1 detection device.
    Returns:
        np.ndarray: The rank-1 detection tensor or its Gramian.
    """
    if len(sz) != 2:
        raise ValueError("rank1_mapping: sz should have length 2.")
    if E.shape[0] != np.prod(sz):
        raise ValueError("rank1_mapping: size(E,1) should be equal to prod(sz).")

    R = E.shape[1]
    if not gramian:
        # Construct rank-1 detection device matrix for each column
        P = np.zeros((np.prod(sz)**2, R * (R + 1) // 2))
        # print((*sz, R))
        E = E.reshape((*sz, R),order='F')
        s, t = np.tril_indices(R)
        sorted_indices = np.argsort(t * R + s)  # Sort by column-major order
        s = s[sorted_indices]
        t = t[sorted_indices]  

        for i in range(len(s)):
            
            P[:, i] = phi(E[:, :, s[i]], E[:, :, t[i]])
        
    else:
        EHE = E.conj().T @ E
        isorth = np.linalg.norm(EHE - np.eye(R), 'fro') / np.linalg.norm(E, 'fro') < np.prod(EHE.shape) * 2 * np.finfo(float).eps

        # Compute the Gramian 0.25 * P.T @ P
        mask = np.tril(np.ones((R, R), dtype=bool)).flatten()

        if isorth:
            P12 = np.ones((R * (R + 1) // 2,))
            P12[mask] = 2
            P12 = np.diag(P12)

        else:
            W = E.conj().T @ E
            P12 = (W[:, None, :, None] * W[None, :, None, :]).reshape(R**2, R**2,order='F')
            P12 += (W[:, None, None, :] * W[None, :, :, None]).reshape(R**2, R**2,order='F')
            P12 = P12[mask][:, mask]

        # Compute terms three and four
        Et = E.reshape(sz[0], sz[1], R,order='F')
        E1 = E.reshape(sz[0], -1,order='F')
        P34 = np.zeros((sz[1], sz[1], R, R))
        for r in range(R):
            X = Et[:, :, r].conj().T @ E1
            P34[:, :, :, r] = X.reshape(sz[1], sz[1], R,order='F')
        P34 = P34.reshape(sz[1]**2, R**2,order='F')
        P34 = P34.conj().T @ P34
        P34 = P34.reshape(R, R, R, R,order='F')
        P34 = -P34.transpose(0, 3, 1, 2).reshape(R**2, R**2,order='F') - P34.transpose(0, 3, 2, 1).reshape(R**2, R**2,order='F')

        # Create final result
        P = P12 + P34[mask][:, mask]

    return P




def cpd3_sgsd(T, U0, options=None):
    """
    Canonical Polyadic Decomposition (CPD) by simultaneous generalized Schur decomposition.
    Args:
        T (np.ndarray): Third-order tensor of shape (I, J, K).
        U0 (list of np.ndarray): Initial factor matrices [U1, U2, U3].
        options (dict): Options for the algorithm:
            - 'MaxIter' (int): Maximum number of iterations (default: 200).
            - 'TolFun' (float): Tolerance for the objective function (default: 1e-4).
    Returns:
        tuple: (U, output)
            - U (list of np.ndarray): Factor matrices [U1, U2, U3].
            - output (dict): Additional information about the optimization process.
    """
    # Default options
    if options is None:
        options = {}
    MaxIter = options.get('MaxIter', 400)
    TolFun = options.get('TolFun', 1e-4)

    # Check the tensor T
    if T.ndim != 3:
        raise ValueError("T must be a third-order tensor (ndim=3).")

    # Check the initial factor matrices U0
    U0 = [np.array(U) for U in U0]
    R = U0[0].shape[1]
    if any(U.shape[1] != R for U in U0):
        raise ValueError("All U0[n] must have the same number of columns (rank R).")
    if sum(R > dim for dim in T.shape) > 1:
        raise ValueError("sum(size(U0[n],2) <= size(T)) should be >= 2.")
    if any(U.shape[0] != dim for U, dim in zip(U0, T.shape)):
        raise ValueError("size(T,n) should equal size(U0[n],1).")

    # Prepermute the tensor so the longest two modes are first
    perm = np.argsort(T.shape)[::-1]
    T = np.transpose(T, perm)
    U0 = [U0[p] for p in perm]
    iperm = np.argsort(perm)

    # Initialize Q and Z
    Q, _ = qr(U0[0], mode='economic')
    Q = Q.conj().T
    Z, _ = qr(np.fliplr(U0[1]), mode='economic')
    Z = np.conj(np.fliplr(Z))

    # Set up the algorithm
    I, J, K = T.shape
    Imax = R - 1 if I == R else R
    Jmin = 2 if J == R else 1
    T1 = T.reshape(I, -1,order = 'F')

    # Simultaneous generalized Schur decomposition
    Rk = np.transpose(
        np.reshape(
            np.reshape(
                np.transpose(
                    np.reshape(Q @ T1, (I, J, K),order = 'F'), (0, 2, 1)
                ).reshape(-1, J,order = 'F') @ Z,
                (I, K, J),order = 'F'
            ),
            (I, J, K),order = 'F'
        ),
        (0, 2, 1)
    )
    output = {
        'fval': [np.linalg.norm(np.tril(np.sum(np.abs(Rk)**2, axis=2), J - R - 1), 'fro')],
        'info': False,
        'iterations': 0
    }

    while not output['info']:
        # Update Q
        Rk = Rk.reshape(I, -1,order = 'F')
        q = np.eye(I,dtype = complex)
        for i in range(Imax):
            u, _, _ = svd(q[i:, :] @ Rk[:, J - R + i::J])
            q[i:, :] = u.conj().T @ q[i:, :]
        Q = q @ Q

        # Update Z
        Rk = q @ Rk
        z = np.eye(J,dtype= complex)
        for j in range(R, Jmin - 1, -1):
            _, _, v = svd(Rk[j - 1, :].reshape(J, K,order = 'F').T @ z[:, :J - R + j])
            z[:, :J - R + j] = z[:, :J - R + j] @ v[:, ::-1]
        Z = Z @ z

        # Apply Q and Z to T
        Rk = np.transpose(
            np.reshape(
                np.reshape(
                    np.transpose(
                        np.reshape(Q @ T1, (I, J, K),order = 'F'), (0, 2, 1)
                    ).reshape(-1, J,order = 'F') @ Z,
                    (I, K, J),order = 'F'
                ),
                (I, J, K),order = 'F'
            ),
            (0, 2, 1)
        )

        # Update the output structure
        output['fval'].append(np.linalg.norm(np.tril(np.sum(np.abs(Rk)**2, axis=2), J - R - 1), 'fro'))
        output['iterations'] += 1
        if abs(output['fval'][-2] - output['fval'][-1]) <= TolFun:
            output['info'] = 1
        if output['iterations'] >= MaxIter:
            output['info'] = 2

    # Reconstruct the factor matrices A, B, and C
    Rk = Rk[:R, -R:, :]
    R1 = np.eye(R, dtype=complex)
    R11 = np.eye(R, dtype=complex)
    for i in range(R - 2, -1, -1):
        for j in range(i + 1, R):
            A1 = Rk[j, j, :]
            A2 = Rk[i, i, :]
            b = Rk[i, j, :]
            for k in range(K):
                p = slice(i + 1, j)
                dk = np.diag(Rk[:, :, k])
                b[k] -= np.sum(R1[i, p].T * dk[p] * R11[p, j])
            rij = np.linalg.lstsq(np.column_stack((A1, A2)), b, rcond=None)[0]
            R1[i, j] = rij[0]
            R11[i, j] = rij[1]
    A = Q[:R, :].conj().T @ R1
    B = (R11 @ Z[:, -R:].conj().T).T
    C = np.reshape(T, (-1, K),order = 'F').T @ np.conj(kr(B, A)) / np.conj((B.conj().T @ B) * (A.conj().T @ A))
    U = [A, B, C]

    # Inverse permute the factor matrices
    U = [U[p] for p in iperm]

    return U, output



def symmetric_null(P, R):
    """
    Compute symmetric matrices in the null space.
    Args:
        P (np.ndarray): Input tall matrix of size (M, R*(R+1)/2).
        R (int): Dimension of the kernel and size of the symmetric matrices.
    Returns:
        np.ndarray: Tensor M of shape (R, R, R), where each frontal slice is a symmetric matrix.
    """
    # Check inputs
    if P.shape[1] != R * (R + 1) // 2:
        raise ValueError("size(P, 2) should equal R*(R+1)/2")

    # Determine null space of P
    _, _, Vp = svd(P, full_matrices=False)

    # Create lower triangular part of M
    M = np.zeros((R**2, R))
    mask = np.tril(np.ones((R, R), dtype=bool))
    M[mask.flatten(order='F'), :] = Vp[:, -R:]


    M = M.reshape((R, R, R),order='F')
    M = M + np.transpose(M, (1, 0, 2))

    return M



def foobi1_T(T, R, **kwargs):
    """
    Fourth-Order-Only Blind Identification of Underdetermined Mixtures.
    Args:
        T (np.ndarray): Input tensor of size (J, J^2).
        R (int): Number of sources.
        **kwargs: Options for the simultaneous diagonalization algorithm.
            - TolFun (float): Tolerance for the objective function.
            - MaxIter (int): Maximum number of iterations.
    Returns:
        np.ndarray: The estimated mixing matrix A of size (J, R).
    """
    # Set options
    options = {
        'TolFun': 1e-5,
        'MaxIter': 200
    }
    options.update(kwargs)

    # Detect number of mixtures
    J = T.shape[0]

    # Compute quadricovariance matrix
    C = T.reshape(J**2, J**2)

    # Compress measurements
    u_C, s_C, _ = svd(C, full_matrices=False)
    u_C = u_C[:, :R]
    s_C = s_C[:R]
    H = u_C * s_C.T


    # Compute Pst through rank-1 mapping
    P = rank1_mapping(H, (J, J))



    # Check uniqueness condition
    if R * (R - 1) > J**2 * (J - 1)**2 / 2:
        scheck = svd(P, compute_uv=False)
        raise ValueError(
            f"R has to satisfy R*(R-1) <= J^2*(J-1)^2/2. "
            f"The smallest singular value that should be nonzero is {scheck[-R]}."
        )

    # Construct auxiliary tensor
    M = symmetric_null(P, R)
 

    # Simultaneous diagonalization
    if R == 1:
        Atilde = (u_C * s_C.T) @ M
    else:
        _, F0 = scipy.linalg.eig(M[:, :, 0], M[:, :, 1])

        F0 = pinv(F0).T
        C0 = (M.reshape(R * R, R).T @ np.conj(kr(F0, F0))) / np.conj((F0.conj().T @ F0)**2)
        F = cpd3_sgsd(M, [F0, F0, C0], options)
        Atilde = (u_C * s_C.T) @ F[0][0]

    # Extract mixing matrix by computing rank-1 approximation for each column
    A = np.empty((J, R), dtype=complex)
    for r in range(R):
        ur, _, _ = svd(Atilde[:, r].reshape(J, J))
        A[:, r] = ur[:, 0]

    return A


#JADE then JADE
def recover_pattern_JADE_JADE(Y,k4_b,k4_f,k2_f,k2_b,r=None,l=None):
    I=k4_b.shape[0]
    if r==None:
        # get flattening of k4_b
        matK4b=k4_b.reshape(I**2,I**2)
        # remove repeat entries from matK
        symind, findsym, symindscale = symmetric_indices(I, 2)
        symindscale = np.sqrt(symindscale)
        findsym = findsym.flatten()
        symind = symind[::-1,:].T @ (I ** np.arange(2))
        sym_matK4b = symindscale.reshape(1, -1) * matK4b[symind][:, symind] * symindscale.reshape(-1, 1)
        D_prime,symV_prime=eig2(sym_matK4b)
        eigtol= 1e-12
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), eigtol)
        # r=min(rank_k4_b,I)
        r=rank_k4_b
    
    
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s = np.real(np.array(np.linalg.pinv(jadeR(Y.T,r))))
    a_s = a_s/np.linalg.norm(a_s,axis = 0)
    
    # get flattening of k4_f,k4_b
    k4_f_copy=k4_f.copy()
    matK4f=k4_f_copy.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    
    D, symV = eig2(sym_matK4f)

    eigtol= 1e-12
    rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)
    if l==None:
        l=rank_k4_f

    rank_k4_f = min(r+l,rank_k4_f)


    D = D[:rank_k4_f]

    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f

    lambda_prime=[]
    
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha

        scalar=(alpha.T @ D1alpha)[0,0]
        if scalar !=0:
            k4_f_copy-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)
            lambda_prime.append(1./scalar)

        matK4f=k4_f_copy.reshape(I**2,I**2)

        sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
        D, symV = eig2(sym_matK4f)
        eigtol= 1e-12
        rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)

        D = D[:rank_k4_f]
        V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
        D1 = np.diagflat(1. / D).T
 
    
   

    B = np.array(np.linalg.pinv(jadeR_from_CM(k4_f_copy,l))).T
    B = B/np.linalg.norm(B,axis = 0)


    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in B.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= B[:,contrastind]
    return a_s, b_s_sorted_contrast



# FOOBI then HTD

def recover_pattern_FOOBI_HTD(k4_b,k4_f,k2_f,k2_b,r=None,l=None):
    I=k4_b.shape[0]
    if r==None:
        # get flattening of k4_b
        matK4b=k4_b.reshape(I**2,I**2)
        # remove repeat entries from matK
        symind, findsym, symindscale = symmetric_indices(I, 2)
        symindscale = np.sqrt(symindscale)
        findsym = findsym.flatten()
        symind = symind[::-1,:].T @ (I ** np.arange(2))
        sym_matK4b = symindscale.reshape(1, -1) * matK4b[symind][:, symind] * symindscale.reshape(-1, 1)
        D_prime,symV_prime=eig2(sym_matK4b)
        eigtol= 1e-12
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), eigtol)
        # r=min(rank_k4_b,I)
        r=rank_k4_b
    
    
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s = np.real(foobi1_T(k4_b,r))
    a_s = a_s/np.linalg.norm(a_s,axis = 0)
    
    # get flattening of k4_f,k4_b
    k4_f_copy=k4_f.copy()
    matK4f=k4_f_copy.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    
    D, symV = eig2(sym_matK4f)

    eigtol= 1e-12
    rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)

    D = D[:rank_k4_f]

    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f

    lambda_prime=[]
    
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha

        scalar=(alpha.T @ D1alpha)[0,0]
        
        k4_f_copy-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)
        
        lambda_prime.append(1./scalar)

        matK4f=k4_f_copy.reshape(I**2,I**2)

        sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
        D, symV = eig2(sym_matK4f)
        eigtol= 1e-12
        rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)

        D = D[:rank_k4_f]
        V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
        D1 = np.diagflat(1. / D).T
    if l==None:
        l=rank_k4_f
    
   

    Diag,B=eig2(k4_f_copy.reshape(I**2,I**2))
    # D_foreground= Diag
    M=k4_f_copy.reshape(I**2,I**2)
    blist=[]
    for n,bpow in enumerate(B.T[:l]):
        bpowmat=bpow.reshape(I,I)
        Diag_prime,eigen=eig2(bpowmat)
        b=eigen[:,0].flatten()
        blist.append(b/np.sum(b*b)**0.5)
        bpow=Diag_prime[0]*khatri_rao_power(b.reshape(-1, 1), 2)
        M-=Diag[n]*bpowmat.reshape(-1,1)@bpowmat.reshape(1,-1)
    blist=np.array(blist).T
    # second_residule=np.sum(M*M)/(I**4)
    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in blist.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= blist[:,contrastind]
    return a_s, b_s_sorted_contrast


# SPM FOOBI 
def recover_pattern_SPM_FOOBI(k4_b,k4_f,k2_b,k2_f,r,l):
    I=k4_b.shape[0]
    if r==None:
        # get flattening of k4_b
        matK4b=k4_b.reshape(I**2,I**2)
        # remove repeat entries from matK
        symind, findsym, symindscale = symmetric_indices(I, 2)
        symindscale = np.sqrt(symindscale)
        findsym = findsym.flatten()
        symind = symind[::-1,:].T @ (I ** np.arange(2))
        sym_matK4b = symindscale.reshape(1, -1) * matK4b[symind][:, symind] * symindscale.reshape(-1, 1)
        D_prime,symV_prime=eig2(sym_matK4b)
        eigtol= 1e-12
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), eigtol)
        # r=min(rank_k4_b,I)
        r=rank_k4_b
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s,lambdas=subspace_power_method(k4_b.copy(),n=4,d=I,r=r)
    # get flattening of k4_f
    matK4f=k4_f.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    D, symV = eig2(sym_matK4f)
    eigtol= 1e-8
    rank_k4_f=D.shape[0] - np.searchsorted(D[::-1], eigtol)
    if l==None:
        l=rank_k4_f
        print('l',l)
    rank_k4_f = min(r+l,rank_k4_f)
    D = D[:rank_k4_f]
    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha
        scalar=(alpha.T @ D1alpha)[0,0]
        if scalar !=0:
            k4_f-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)

    # recover b_i vectors (foreground patterns) and prevent repetitive vectors
    b_s =np.real(foobi1_T(k4_f,l))
    b_s = b_s/np.linalg.norm(b_s,axis = 0)
    # construct a matrix with columns Vect(b^{\otimes 2})
    vpowlist=[]
    for a in a_s.T:
        vpowlist.append(khatri_rao_power(a.reshape(-1, 1), 2).reshape(-1,1))
    for b in b_s.T:
        vpowlist.append(khatri_rao_power(b.reshape(-1, 1), 2).reshape(-1,1))
    
    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in b_s.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@ k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= b_s[:,contrastind]
    return a_s,b_s_sorted_contrast


# FOOBI FOOBI 
def recover_pattern_FOOBI_FOOBI(k4_b,k4_f,k2_b,k2_f,r,l):
    I=k4_b.shape[0]
    if r==None:
        # get flattening of k4_b
        matK4b=k4_b.reshape(I**2,I**2)
        # remove repeat entries from matK
        symind, findsym, symindscale = symmetric_indices(I, 2)
        symindscale = np.sqrt(symindscale)
        findsym = findsym.flatten()
        symind = symind[::-1,:].T @ (I ** np.arange(2))
        sym_matK4b = symindscale.reshape(1, -1) * matK4b[symind][:, symind] * symindscale.reshape(-1, 1)
        D_prime,symV_prime=eig2(sym_matK4b)
        eigtol= 1e-12
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), eigtol)
        # r=min(rank_k4_b,I)
        r=rank_k4_b
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s =np.real(foobi1_T(k4_b,r))
    a_s = a_s/np.linalg.norm(a_s,axis = 0)
    # get flattening of k4_f
    matK4f=k4_f.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    D, symV = eig2(sym_matK4f)
    eigtol= 1e-8
    rank_k4_f=D.shape[0] - np.searchsorted(D[::-1], eigtol)
    if l==None:
        l=rank_k4_f-r
        print('l',l)
    rank_k4_f = min(r+l,rank_k4_f)
    D = D[:rank_k4_f]
    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha
        scalar=(alpha.T @ D1alpha)[0,0]
        if scalar !=0:
            k4_f-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)

    # recover b_i vectors (foreground patterns) and prevent repetitive vectors
    b_s =np.real(foobi1_T(k4_f.copy(),l))
    b_s = b_s/np.linalg.norm(b_s,axis = 0)
    # construct a matrix with columns Vect(b^{\otimes 2})
    vpowlist=[]
    for a in a_s.T:
        vpowlist.append(khatri_rao_power(a.reshape(-1, 1), 2).reshape(-1,1))
    for b in b_s.T:
        vpowlist.append(khatri_rao_power(b.reshape(-1, 1), 2).reshape(-1,1))
    
    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in b_s.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@ k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= b_s[:,contrastind]
    return k4_f,a_s,b_s_sorted_contrast





def recover_pattern_FastICA_HTD(Y,k4_b,k4_f,k2_f,k2_b,r=None,l=None):
    I=k4_b.shape[0]
    if r==None:
        # get flattening of k4_b
        matK4b=k4_b.reshape(I**2,I**2)
        # remove repeat entries from matK
        symind, findsym, symindscale = symmetric_indices(I, 2)
        symindscale = np.sqrt(symindscale)
        findsym = findsym.flatten()
        symind = symind[::-1,:].T @ (I ** np.arange(2))
        sym_matK4b = symindscale.reshape(1, -1) * matK4b[symind][:, symind] * symindscale.reshape(-1, 1)
        D_prime,symV_prime=eig2(sym_matK4b)
        eigtol= 1e-12
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), eigtol)
        # r=min(rank_k4_b,I)
        r=rank_k4_b
    
    
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    ica = FastICA(n_components=r)
    S_ = ica.fit_transform(Y)  # Recovered signals
    a_s = ica.mixing_
    a_s = a_s/np.linalg.norm(a_s,axis = 0)

    # get flattening of k4_f,k4_b
    k4_f_copy=k4_f.copy()
    matK4f=k4_f_copy.reshape(I**2,I**2)

    # remove repeat entries from matK
    symind, findsym, symindscale = symmetric_indices(I, 2)
    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (I ** np.arange(2))
    sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
    
    D, symV = eig2(sym_matK4f)

    eigtol= 1e-12
    rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)
    if l==None:
        l=rank_k4_f
        print('l',l)
    rank_k4_f = min(r+l,rank_k4_f)

    D = D[:rank_k4_f]

    V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
    D1 = np.diagflat(1. / D).T

    # learn coefficients of a_i in k4_f

    lambda_prime=[]
    
    for a in a_s.T:
        apow = khatri_rao_power(a.reshape(-1, 1), 2)
        alpha = (apow.T @ V).T
        D1alpha = D1 @ alpha

        scalar=(alpha.T @ D1alpha)[0,0]
        if scalar !=0:
            k4_f_copy-=(1. / scalar)*generate_lowrank_tensor(a.reshape(-1,1),4)
        
            lambda_prime.append(1./scalar)

        matK4f=k4_f_copy.reshape(I**2,I**2)

        sym_matK4f = symindscale.reshape(1, -1) * matK4f[symind][:, symind] * symindscale.reshape(-1, 1)
        D, symV = eig2(sym_matK4f)
        eigtol= 1e-12
        rank_k4_f=D.shape[0] - np.searchsorted(abs(D[::-1]), eigtol)

        D = D[:rank_k4_f]
        V = (symV[:, :rank_k4_f] / symindscale.reshape(-1, 1))[findsym, :]
        D1 = np.diagflat(1. / D).T
    if l==None:
        l=rank_k4_f
        print('l',l)
    
   

    Diag,B=eig2(k4_f_copy.reshape(I**2,I**2))
    # D_foreground= Diag
    M=k4_f_copy.reshape(I**2,I**2)
    blist=[]
    for n,bpow in enumerate(B.T[:l]):
        bpowmat=bpow.reshape(I,I)
        Diag_prime,eigen=eig2(bpowmat)
        b=eigen[:,0].flatten()
        blist.append(b/np.sum(b*b)**0.5)
        bpow=Diag_prime[0]*khatri_rao_power(b.reshape(-1, 1), 2)
        M-=Diag[n]*bpowmat.reshape(-1,1)@bpowmat.reshape(1,-1)
    blist=np.array(blist).T
    # second_residule=np.sum(M*M)/(I**4)
    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in blist.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= blist[:,contrastind]
    return a_s, b_s_sorted_contrast


# SPM SPM separately

def recover_pattern_separate(k4_b,k4_f,k2_b,k2_f,r,l):
    I=k4_b.shape[0]
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s,lambdas=subspace_power_method(k4_b.copy(),n=4,d=I,r=r)
    

    a_b_s,lambdas_prime = subspace_power_method(k4_f.copy(),n=4,d=I,r=r+l)
    background_index_list = []
    for a in a_s.T:
        possible_indices = list(set(range(r+l))-set(background_index_list))

        inner_products = abs(a.reshape(1,-1)@a_b_s[:,possible_indices])
        # print(inner_products,max(inner_products))
        if np.max(inner_products)>0.9:
            index = possible_indices[np.argmax(inner_products)]
            background_index_list.append(index)
    b_s = [a_b_s[:,i] for i in range(r+l) if i not in background_index_list ]
    b_s = np.array(b_s).T

    # rank b_i according to their variance ratio
    contrastvarlist=[]
    fore_varlist=[]
    back_varlist=[]
    for b in b_s.T:
        b=b.reshape(-1,1)
        fore_var=(b.T@ k2_f @ b).flatten()
        back_var=(b.T@ k2_b @ b).flatten()
        # varlist.append(fore_var)
        contrastvarlist.append(fore_var/back_var)
        fore_varlist.append(fore_var)
        back_varlist.append(back_var)

    contrastind=(-np.array(contrastvarlist)).flatten().argsort().tolist()
    b_s_sorted_contrast= b_s[:,contrastind]

    return a_s, b_s_sorted_contrast


