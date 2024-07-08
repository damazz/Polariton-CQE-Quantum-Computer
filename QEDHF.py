import numpy as np
import numpy.linalg as LA
from scipy.linalg import sqrtm
from pyscf import gto, scf, ao2mo, fci
from functools import reduce, cache
#from QED.tools import center
#import sparse as sp
template = "{} {} {} {}; "
def center(mol):
    coords = mol.atom_coords()
    charge = mol.atom_charges()
    species = [mol.atom_symbol(i) for i in range(len(charge))]
    vec = (1/np.sum(charge)*np.tensordot(charge,coords,axes = 1))
    coords-=vec
    string = ""
    for i,spec in enumerate(species):
        string+=template.format(spec,coords[i,0],coords[i,1],coords[i,2])
    molt = gto.M(atom = string,
            basis = mol.basis,
            unit = "B",
            spin = mol.spin,
            )
    return molt 


class _qedhf:
    def __init__(self,hfobj,lamb,omega = 1):
        mol = center(hfobj.mol)
        self.mol = mol
        if not hfobj.converged:
            hfobj.kernel()
        self.converged = False
        self.eig = hfobj.eig
        self.get_occ = hfobj.get_occ
        self.omega_cav = omega
        self.mo_occ = hfobj.mo_occ
        self.HF_en = hfobj.e_tot
        self.e_tot = hfobj.e_tot
        self.E_nuc = mol.energy_nuc()
        self.nuc_dip = np.tensordot(mol.atom_charges(),mol.atom_coords(),axes = 1)
        self.lamb = lamb
        self.hcore = mol.intor('int1e_kin')+mol.intor('int1e_nuc')
        self.S = mol.intor('int1e_ovlp')
        self.A = LA.inv(sqrtm(self.S))
        self.eri = mol.intor('int2e') 
        self.D = hfobj.make_rdm1()
        self.dpmat = -mol.intor('int1e_r')
        self.qdmat = -mol.intor('int1e_rr').reshape(3,3,self.dpmat.shape[-1],self.dpmat.shape[-1])
        
        self.dpop = np.tensordot(self.lamb,self.dpmat,axes = 1)
        self.qdop = np.tensordot(np.outer(lamb,lamb),self.qdmat,axes =((0,1),(0,1)))

    def make_rdm1(self):
        return self.D
    

class qedrhf(_qedhf):
    def __init__(self,hfobj,lamb,omega = 1):
        super().__init__(hfobj,lamb,omega)
        assert(self.mol.nelec[0]==self.mol.nelec[1])

    def kernel(self):
        eold = 0
        while np.abs(eold-self.e_tot)>1E-12:
            eold = self.e_tot
            J = np.einsum("pqrs,rs->pq",self.eri,self.D)
            K = np.einsum("prqs,rs->pq",self.eri,self.D)
            M = np.einsum("pq,rs,rs->pq",self.dpop,self.dpop,self.D)
            N = np.einsum("pr,qs,rs->pq",self.dpop,self.dpop,self.D)

            self.onebody = np.copy(self.hcore)
            self.onebody -= .5*self.qdop
            self.onebody -= self.dpop*np.tensordot(self.dpop,self.D, axes = 2)
            self.twobody = (J-1/2*K)+(M-1/2*N)
            self.F = self.onebody+self.twobody
            self.e_tot = np.trace(np.matmul(self.onebody+1/2*self.twobody,self.D))
            self.e_tot += self.E_nuc
            self.e_tot += 1/2*np.tensordot(self.dpop,self.D, axes = 2)**2
            Fp = self.A.dot(self.F).dot(self.A)
            e, Cp = LA.eigh(Fp)
            C = self.A.dot(Cp)
            Cocc = C[:,:self.mol.nelec[0]]
            self.D_old = np.copy(self.D)
            self.D = 2*np.tensordot(Cocc,Cocc,axes = ((1),(1)))
        print("converged QED Energy:",self.e_tot)
        self.converged = True
        self.mo_e, self.mo_c = self.eig(self.F,self.S)
        self.mo_occ = self.get_occ(self.mo_e,self.mo_c)
        twobody = np.copy(self.eri)+np.tensordot(self.dpop,self.dpop,axes = 0)
        self.twobody_mo = ao2mo.kernel(twobody,self.mo_c,aosym = 's1').reshape(tuple((self.mo_occ.shape[0] for i in range(4))))
        self.dipexp = np.tensordot(self.dpop,self.D,axes=2)

class qeduhf(_qedhf):
    def __init__(self,hfobj,lamb,omega = 1):
        super().__init__(hfobj,lamb,omega)

    def kernel(self):
        eold = 0
        while np.abs(eold-self.e_tot)>1E-12:
            eold = self.e_tot
            J = np.einsum("pqrs,lrs->lpq", self.eri,self.D)
            K = np.einsum("prqs,lrs->lpq",self.eri,self.D)
            M = np.einsum("pq,rs,lrs->lpq",self.dpop,self.dpop,self.D)
            N = np.einsum("pr,qs,lrs->lpq",self.dpop,self.dpop,self.D)

            self.onebody = np.array([np.copy(self.hcore),np.copy(self.hcore)])
            self.onebody -= .5*self.qdop
            self.onebody -= np.tensordot(np.tensordot(self.dpop,self.D, axes = ((0,1),(1,2))),self.dpop,axes = 0)
            self.twobody = (J[0]+J[1]-K)+(M[0]+M[1]-N)
            self.F = self.onebody+self.twobody
            self.e_tot = np.trace(np.matmul(self.onebody[0]+1/2*self.twobody[0],self.D[0]))
            self.e_tot += np.trace(np.matmul(self.onebody[1]+1/2*self.twobody[1],self.D[1]))
            self.e_tot += self.E_nuc
            self.e_tot += 1/2*np.sum(np.tensordot(self.dpop,self.D, axes = ((0,1),(1,2)))**2)
            Fp = np.zeros(self.F.shape)
            Fp[0] = self.A.dot(self.F[0]).dot(self.A)
            Fp[1] = self.A.dot(self.F[1]).dot(self.A)
            e, Cp, = LA.eig(Fp)
            C = np.zeros(Cp.shape)
            C[0] = self.A.dot(Cp[0][:,::int((-1)**(e[0,0]>e[0,-1]))])
            C[1] = self.A.dot(Cp[1][:,::int((-1)**(e[1,0]>e[1,-1]))])
            Cocca = C[0,:,:self.mol.nelec[0]]
            Coccb = C[1,:,:self.mol.nelec[1]]
            self.D_old = np.copy(self.D)
            self.D[0] = np.tensordot(Cocca,Cocca,axes = ((1),(1)))
            self.D[1] = np.tensordot(Coccb,Coccb,axes = ((1),(1)))
        print("converged QED Energy:",self.e_tot)
        self.converged = True
        self.mo_e, self.mo_c = self.eig(self.F,self.S)
        self.mo_occ = self.get_occ(self.mo_e,self.mo_c)
#        twobody = np.copy(self.eri)+np.tensordot(self.dpop,self.dpop,axes = 0)
#        self.twobody_mo = ao2mo.kernel(twobody,self.mo_c,aosym = 's1').reshape(tuple((self.mo_occ.shape[0] for i in range(4))))
        self.dipexp = np.tensordot(self.dpop,self.D,axes=((0,1),(1,2)))



def qedhf(hfobj,lamb,omega = 1):
    if isinstance(hfobj,scf.hf.RHF):
        return qedrhf(hfobj,lamb,omega)
    elif isinstance(hfobj,scf.uhf.UHF):
        return qeduhf(hfobj,lamb,omega)



    

