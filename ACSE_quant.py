import numpy as np
import numpy.linalg as LA
from QEDHF import qedhf
from ham_ops import residual_H2 as residual
from ham_ops import _prep
from itertools import product
from QEDFCI import qedfci

from scipy.linalg import expm, logm
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


class ACSE(residual):
    def __init__(self,qedhf,nb,resstr):
        if qedhf is not None:
            super().__init__(qedhf,nb,resstr)
        else:
            print("Need qedhf object")
        self.ket = qedfci(qedhf).make_HF(nb)
        self.e = []
        self.rdm = []
        self.shots = 8192
        self.Ulist = []
        
    def run(self,p, ket = None):
        if ket is None:
            ket = self.ket
        e0 = LA.eigvalsh(self.H_fci)[0]
        U = np.eye(self.ket.shape[0])
        for i in range(p):
            nq = self.num_qb+self.num_qf
            qc = QuantumCircuit(nq,nq)
            for U in self.Ulist:
                qc.append(U,range(self.nq))
            self.e.append(np.real(self.energy(qc = qc)))
            self.rdm.append(self.meas_rdm(qc = qc))
            tres, ops = self.resops(qc = qc)

            if len(ops)==2:
                p_s = 5
                guess = np.linspace(.01,.4,p_s)
                guess2 = np.linspace(.0001,.5,5)
                guess = np.array(list(product(guess,guess2)))
                out = self.surface(guess,ops,qc)
                #print(out)
            elif len(ops)==1:
                p_s = 9
                guess = np.linspace(.001,1,p_s).reshape(p_s,1)
                out = self.surface(guess,ops,qc)
                #print(out)
            u = self._func2_time(out,ops)
            self.Ulist.extend(u)

    def surface(self,rang,ops,qc):
        e = []
        for i in rang:
            tops = [j*ops[indj] for indj,j in enumerate(i)]
            qct = qc.copy()
            for op in tops :
                qct.append(PauliEvolutionGate(self.mat_to_pauli(op),time = 1),range(self.nq))
            e.append(np.real(self.energy(qc = qct)))
        return rang[np.argmin(e)]

    def _func2_time(self,c,ops):
        tops = [j*ops[indj] for indj,j in enumerate(c)]
        qct = []
        for op in tops:
            qct.append(PauliEvolutionGate(self.mat_to_pauli(op),time = 1))
        return qct

    def mat_to_pauli(self,mat):
        rep = np.tensordot(_prep(self.nq),mat,axes = 1).trace(axis1 = -1,axis2 = -2)
        dic = {0:"I",1:"X",2:"Y",3:"Z"}
        lets = np.argwhere(rep.round(8)).tolist()
        lets = [[dic[l] for l in k] for k in lets]
        lets = ["".join(k) for k in lets]
        obj = SparsePauliOp(lets, [(1.j*rep[tuple(c)]).real for c in np.argwhere(rep.round(8))])
        return obj

