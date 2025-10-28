import numpy as np

def calculate_wt_max_unit(patch, wts, pool_size):
    pmax = np.einsum("ijk,k->ijk",np.ones_like(patch),np.max(np.max(patch,axis=0),axis=0))
    indexes = (patch-pmax)==0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0/np.einsum("mnc->c",indexes)
    indexes = np.einsum("ijk,k->ijk",indexes,indexes_norm)
    out = np.einsum("ijk,k->ijk",indexes,wts)
    return out
