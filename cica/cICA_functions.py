import numpy as np
from .helper_functions import khatri_rao_power, tucker_product, symmetric_indices
from .SPM import subspace_power_method, generate_lowrank_tensor
# eigendecomposition that sorts by absolute eigenvalues
def eig2(a):
    D, V = np.linalg.eigh(a)
    #ind = (-D).argsort(axis=-1)
    ind = (-abs(D)).argsort(axis=-1)
    D = np.take_along_axis(D, ind, axis=-1)
    V = np.take_along_axis(V, np.expand_dims(ind,-2), axis=-1)
    return D, V



# function to compute reconstruction error
def return_residual(T,a_s,lambda_s):
    #a_s size (I,r), b_s size (r,)
    I=T.shape[0]
    T_prime=T.copy()
    r=a_s.shape[1]
    for i in range(r):
        T_prime-=generate_lowrank_tensor(a_s[:,i].reshape(-1,1),4)*lambda_s[i]
    return np.sum(T*T)




# code for proportional cICA via HTD
def HTD(k4_b,k4_f,k2_b,k2_f,gamma,l):
    T=(k4_f-gamma*k4_b).copy()
    I=T.shape[0]
    Diag,B=eig2(T.reshape(I**2,I**2))
    M=T.reshape(I**2,I**2)
    blist=[]
    for n,bpow in enumerate(B.T[:l]):
        bpowmat=bpow.reshape(I,I)
        Diag_prime,eigen=eig2(bpowmat)
        b=eigen[:,0].flatten()
        blist.append(b/np.sum(b*b)**0.5)
        bpow=Diag_prime[0]*khatri_rao_power(b.reshape(-1, 1), 2)
        M-=Diag[n]*bpowmat.reshape(-1,1)@bpowmat.reshape(1,-1)
    blist=np.array(blist).T

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
    return b_s_sorted_contrast,np.array(contrastvarlist).flatten()[contrastind]



# code for cICA via SPM then HTD
def recover_pattern_tensor_eigen(k4_b,k4_f,k2_f,k2_b,r=None,l=None,verbose = False,step_max=1):
    I=k4_b.shape[0]

    # select r if not given
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
        
        rank_k4_b=D_prime.shape[0]-np.searchsorted(abs(D_prime[::-1]), 1e-3)
        r=rank_k4_b
    
    # recover the a_i vectors (backgroud patterns) & prevent repetitive vectors
    a_s,lambdas=subspace_power_method(k4_b,n=4,d=I,r=r)
    first_residule=return_residual(k4_b,a_s,lambdas)/(I**4)
    def returnmindistancebewteenvectors(cols):
        def distancebewteenvectors(v1,v2):
            v1=v1.reshape(-1,1)
            v2=v2.reshape(-1,1)
            M=v1@np.transpose(v1)-v2@np.transpose(v2)
            return np.sum(M*M)
        lens=cols.shape[1]
        error=1
        for i in range(lens):
            for j in range(i+1,lens):
                error=min(error,max(distancebewteenvectors(cols[:,i],cols[:,j]),distancebewteenvectors(cols[:,i],-cols[:,j])))
        return error
    step=0
    if step_max==None:
        step_max=100
    while returnmindistancebewteenvectors(a_s)<0.1 and step<step_max:
        a_s,lambdas=subspace_power_method(k4_b,n=4,d=I,r=r)
        step+=1
    first_residule=return_residual(k4_b,a_s,lambdas)/(I**4)
    if verbose:
        print(f"First stage residual: {first_residule}")
    
    # learn coefficients of a_i in k4_f
    k4_f_copy=k4_f.copy()
    matK4f=k4_f_copy.reshape(I**2,I**2)
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
    
    # define l if not given
    if l==None:
        l=rank_k4_f
    
    # recover b_i vectors (foreground patterns) via HTD
    Diag,B=eig2(k4_f_copy.reshape(I**2,I**2))
    D_foreground=Diag
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
    second_residule=np.sum(M*M)/(I**4)
    if verbose:
        print(f"Second stage residual: {second_residule}")
    
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
    return a_s,b_s_sorted_contrast,np.array(contrastvarlist).flatten()[contrastind]



# function to compute similarity measures between columns of two matrices
# up to permutation and sign flipping
def similarity_measures_withpermutation(B,A):
    J=A.shape[1]
    columnlist=[]
    for i in range(J):
        v=A[:,i]
        dislist=[]
        signlist=[]
        for j in range(B.shape[1]):
            u=abs(v-B[:,j])
            uprime=abs(v+B[:,j])
            sign=1
            if np.sum(u**2)<np.sum(uprime**2):
                dislist.append(np.sum(u**2))
                signlist.append(sign)
            else:
                sign=-1
                dislist.append(np.sum(uprime**2))
                signlist.append(sign)
        a=min(dislist)
        index=dislist.index(a)
        sign=signlist[index]
        columnlist.append(sign*B[:,index])
        B=np.delete(B,index,1)
    
    permutedB=np.transpose(np.vstack(tuple(columnlist)))
    C= permutedB-A
    relfroberror=(np.sum(C*C)/J)**0.5
    froberror=(np.sum(C*C))**0.5
    cosine_list= np.sum(permutedB*A,axis=0)
    cosine_similarity=np.mean(cosine_list)
    return cosine_list,cosine_similarity,relfroberror,froberror



# HTD decomposition only
def HTD_decomp(T,l):
    I=T.shape[0]
    Diag,B=eig2(T.reshape(I**2,I**2))
    M=T.reshape(I**2,I**2)
    blist=[]
    for n,bpow in enumerate(B.T[:l]):
        bpowmat=bpow.reshape(I,I)
        Diag_prime,eigen=eig2(bpowmat)
        b=eigen[:,0].flatten()
        blist.append(b/np.sum(b*b)**0.5)
        bpow=Diag_prime[0]*khatri_rao_power(b.reshape(-1, 1), 2)
        M-=Diag[n]*bpowmat.reshape(-1,1)@bpowmat.reshape(1,-1)
    blist=np.array(blist).T

    return blist