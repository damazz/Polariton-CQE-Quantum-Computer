import numpy as np
import numpy.linalg as LA
from scipy.linalg import sqrtm
from pyscf import gto, scf, ao2mo, fci
from functools import reduce, cache
from QEDHF import qedhf, qedrhf
from itertools import product
from math import comb


class qedfci:
    def __init__(self,qedhf):
        assert isinstance(qedhf,qedrhf)
        self.mol = qedhf.mol
        self.mo_c = qedhf.mo_c
        self.mo_occ = qedhf.mo_occ
        self.hcore = qedhf.hcore
        self.twobody_mo = qedhf.twobody_mo
        self.dpop = qedhf.dpop
        self.qdop = qedhf.qdop
        self.dipexp = qedhf.dipexp
        self.omega_cav = qedhf.omega_cav
        self.lamb = qedhf.lamb
        self._prep_dic = {
            "sd": self._prep_sd,
            }

    def fci(self, nb, strr = "sd"):
        nb+=1
        return self.assemble(nb, strr)

    def assemble(self,bosnum, strr = "sd"):
        hcore, dipole, qpole, twobody, Id = self._prep_dic[strr]()
        ovlpmat = np.eye(bosnum)
        boscav = self.omega_cav * np.diag(np.arange(bosnum))
        dipole_coup = np.sqrt(np.vstack([np.diag(np.arange(bosnum))[1:],np.zeros(bosnum)]))
        dipole_coup += dipole_coup.T
        dipole_coup *= np.sqrt(self.omega_cav/2)

        H_fci = np.kron(ovlpmat,(twobody+hcore-1/2*qpole-dipole*self.dipexp)+Id*self.dipexp**2)
        H_fci-= np.kron(dipole_coup, dipole-Id*self.dipexp)
        H_fci+= np.kron(boscav, Id)
        return H_fci + np.kron(np.eye(bosnum),Id)*self.mol.energy_nuc()

    #Preparation for FCI in slater determinate basis
    @cache
    def _prep_sd(self):
        onebody_hcore = np.einsum('pi,pq,qj->ij', self.mo_c, self.hcore, self.mo_c)
        onebody_dpop = np.einsum('pi,pq,qj->ij', self.mo_c, self.dpop, self.mo_c)
        onebody_qdop = np.einsum('pi,pq,qj->ij', self.mo_c, self.qdop, self.mo_c)
        self.onebody_hcore = fci.direct_spin0.pspace(onebody_hcore,np.zeros(self.twobody_mo.shape),len(self.mo_occ),tuple((self.mol.nelec[0],self.mol.nelec[0])),np = 400000)[1]
        self.onebody_dpop = fci.direct_spin0.pspace(onebody_dpop,np.zeros(self.twobody_mo.shape),len(self.mo_occ),tuple((self.mol.nelec[0],self.mol.nelec[0])),np = 400000)[1]
        self.onebody_qdop = fci.direct_spin0.pspace(onebody_qdop,np.zeros(self.twobody_mo.shape),len(self.mo_occ),tuple((self.mol.nelec[0],self.mol.nelec[0])),np = 400000)[1]
        self.twobody_fci = fci.direct_spin0.pspace(np.zeros(onebody_hcore.shape),self.twobody_mo,len(self.mo_occ),tuple((self.mol.nelec[0],self.mol.nelec[0])),np = 400000)[1]
        return (self.onebody_hcore, self.onebody_dpop, self.onebody_qdop,
                self.twobody_fci, np.eye(self.twobody_fci.shape[-1]))

    def make_HF(self,nb,strr='sd'):
        if strr == "sd":
            n = np.sum(self.mo_occ)//2
            ket = np.zeros((comb(len(self.mo_occ),int(n))**2))
            ket[0]=1
        return ket 

