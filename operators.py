from math import comb
from itertools import product
import numpy as np
import sparse as sp

class operators(object):
    def __init__(self,mol):
        self.mol = mol
        self.lg = self.mol.intor('int1e_kin').shape[0]
        self.size = (comb(self.lg,self.mol.nelec[0]))*(comb(self.lg,self.mol.nelec[1]))
        self._mappings()
        self.rdm1ops()
        self.rdm2ops()


    def _mappings(self):
        self.states_d = {}
        self.states_l = np.array(list(product(combination(self.lg,self.mol.nelec[0]),combination(self.lg,self.mol.nelec[1])+self.lg))).reshape(comb(self.lg,self.mol.nelec[0])*comb(self.lg,self.mol.nelec[1]),-1)
        for i,st in enumerate(self.states_l):
            s = "".join([str(int(ele)) for ele in st])
            self.states_d[s] = i
        
    def rdm1ops(self):
        a_ij_index = []
        a_ij_val = []
        for spin in range(2):
            for i in range(self.lg):
                for j in range(self.lg):
                    for ind, st in enumerate(self.states_l):
                        try:
                            temp = st.tolist()
                            ind1 = temp.index(j+spin*self.lg)
                            temp.remove(j+spin*self.lg)
                            temp.append(i+spin*self.lg)
                            temp.sort()
                            ind2 = temp.index(i+spin*self.lg)
                            s = "".join([str(int(ele)) for ele in temp])
                            a_ij_index.append([spin,i,j,self.states_d[s],ind])
                            a_ij_val.append((-1)**(ind1+ind2))
                        except ValueError:
                            pass
                        except KeyError:
                            pass
        self.a_ij = sp.COO(np.array(a_ij_index).T,data = a_ij_val,shape = (2,self.lg,self.lg,self.size,self.size))

    def rdm2ops(self):
        a_ijkl_index = []
        a_ijkl_val = []
        #alpha/alpha
        for prod in product(range(self.lg),repeat = 4):
                i,j,k,l = prod
                if i!=j and k!=l:
                    for ind, st in enumerate(self.states_l):
                        try:
                            temp = st.tolist()
                            ind1 = temp.index(l)
                            ind2 = temp.index(k)
                            temp.remove(l)
                            temp.remove(k)
                            temp.append(j)
                            temp.append(i)
                            temp.sort()
                            ind3 = temp.index(j)
                            ind4 = temp.index(i)
                            s = "".join([str(int(ele)) for ele in temp])
                            a_ijkl_index.append([0,i%self.lg,j%self.lg,k%self.lg,l%self.lg,self.states_d[s],ind])
                            a_ijkl_val.append((-1)**(ind1+ind2+(l>k)+ind3+ind4+(i>j)))
                        except ValueError:
                            pass
                        except KeyError:
                            pass

        #alpha/beta
        for prod in product(range(self.lg),range(self.lg,2*self.lg),range(self.lg),range(self.lg,2*self.lg)):
                i,j,k,l = prod
                #if i!=j and k!=l:
                for ind, st in enumerate(self.states_l):
                        try:
                            temp = st.tolist()
                            ind1 = temp.index(l)
                            ind2 = temp.index(k)
                            temp.remove(l)
                            temp.remove(k)
                            temp.append(j)
                            temp.append(i)
                            temp.sort()
                            ind3 = temp.index(j)
                            ind4 = temp.index(i)
                            s = "".join([str(int(ele)) for ele in temp])
                            a_ijkl_index.append([1,i%self.lg,j%self.lg,k%self.lg,l%self.lg,self.states_d[s],ind])
                            a_ijkl_val.append((-1)**(ind1+ind2+(l>k)+ind3+ind4+(i>j)))
                        except ValueError:
                            pass
                        except KeyError:
                            pass

        #beta/beta
        for prod in product(range(self.lg,2*self.lg),repeat = 4):
                i,j,k,l = prod
                if i!=j and k!=l:
                    for ind, st in enumerate(self.states_l):
                        try:
                            temp = st.tolist()
                            ind1 = temp.index(l)
                            ind2 = temp.index(k)
                            temp.remove(l)
                            temp.remove(k)
                            temp.append(j)
                            temp.append(i)
                            temp.sort()
                            ind3 = temp.index(j)
                            ind4 = temp.index(i)
                            s = "".join([str(int(ele)) for ele in temp])
                            a_ijkl_index.append([2,i%self.lg,j%self.lg,k%self.lg,l%self.lg,self.states_d[s],ind])
                            a_ijkl_val.append((-1)**(ind1+ind2+(l>k)+ind3+ind4+(i>j)))
                        except ValueError:
                            pass
                        except KeyError:
                            pass
        self.a_ijkl = sp.COO(np.array(a_ijkl_index).T,data = a_ijkl_val,shape = (3,self.lg,self.lg,self.lg,self.lg,self.size,self.size))
    
    def make_1rdms(self,ket,nb):
        nb+=1
        rho = np.outer(ket.conj(),ket).reshape(nb,self.size,nb,self.size).trace(axis1 = 0,axis2 = 2)
        return np.tensordot(self.a_ij,rho,axes = 2)

import numba
@numba.jit(nopython = True , nogil=True, cache = True)
def comb(n,k):
    if n<k:
        return 0 
    elif n==k:
        return 1
    c = n-k
    pa = np.int64(1)
    pb = np.int64(1)
    if k > c:
        for i in range(k+1,n+1):
            pa *= i 
        for i in range(1,c+1):
            pb *= i
    else:
        for i in range(c+1,n+1):
            pa *= i
        for i in range(1,k+1):
            pb *= i 
    return pa // pb

@numba.jit(nopython = True, nogil = True, cache = True)
def rindexer(num,N,r):
    out = np.arange(N,dtype = np.int64)
    k = -1
    for i in range(r):
        if num==0:
            return out
        elif num//comb(r-i,N)>0:
            out[k]=r-i
            num-=comb(r-i,N)
            N-=1
            k-=1
    return out

@numba.jit(nopython = True, nogil = True,parallel = True, cache = True)
def combination(r,N):
    out = np.zeros((comb(r,N),N),dtype = np.int64)
    for i in numba.prange(comb(r,N)):
        out[i] = rindexer(i,N,r)
    return out

