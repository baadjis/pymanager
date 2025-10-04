from  sklearn.decomposition  import  KernelPCA
from  sklearn.decomposition  import FastICA
import numpy as np
normalize=lambda x: (x-x.mean())/x.std()
fractions=lambda x: x/x.sum()

def pca(data,n_components=1):
    pc=KernelPCA(n_components =n_components).fit(data.apply(normalize))
    pca_components=pc.transform(-data)
    weights=fractions(pc.eigenvalues_)
    return np.dot(pca_components,weights)

def ica(data,n_components=2):
    ic=FastICA(n_components=n_components)
    
    s=ic.fit_transform(data.apply(normalize))
    print(ic.components_.shape)
    w=ic.mixing_
    
    wg=fractions(w)
    r=wg@s.T
    print(data.shape)
    print(s.shape)
    print(wg.shape)
    print(r.shape)
    return s