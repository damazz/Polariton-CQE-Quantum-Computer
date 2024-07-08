import numpy as np
import numpy.linalg as LA
from QEDHF import qedhf,qedrhf
from QEDFCI import qedfci

from itertools import combinations, product
from functools import reduce
from operators import operators
import sparse
from scipy.linalg import expm

from qiskit import QuantumCircuit

from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from functools import cache, reduce
sig = np.zeros((4,2,2),dtype = complex)
sig[0] = 1/2*np.eye(2)
sig[1] = 1/2*np.array([[0,1],[1,0]])
sig[2] = 1/2*np.array([[0,-1.j],[1.j,0]])
sig[3] = 1/2*np.array([[1,0],[0,-1]])
@cache
def _prep(n):
    out = np.zeros((4**n,2**n,2**n),dtype = complex)
    for ind, perm in enumerate(product(sig, repeat = n)):
        out[ind] = reduce(np.kron,perm)
    out = sparse.COO.from_numpy(out.reshape(tuple((4 for i in range(n)))+tuple((2**n,2**n))))
    return out


def graph_red(str_list):
    set_meas = set()
    for k in str_list:
        set_meas.add(tuple(k))
    list_meas = list(set_meas)
    tuple_out = []
    for ind, i in enumerate(list_meas):
        tuple_out.append(tuple([tuple(i)]))
    return tuple_out

def num_to_let(nums):
    dic = {0:'I',
           1:'X',
           2:'Y',
           3:'Z'}
    out = []
    for i in nums:
        tl = []
        for num in i:
            tl.append(dic[num])
        out.append("".join(tl))
    return out
    

class residual_H2:
    def __init__(self,qedhf_obj,bosnum,resstr):
        if isinstance(qedhf_obj,qedrhf):
            qedfci_obj = qedfci(qedhf_obj)
        else:
            print("Please supply QED-RHF object")
            exit()
        self.resstr = resstr
        self.H_fci = qedfci_obj.fci(bosnum,'sd').round(12)

        self.nq = int(np.log2(self.H_fci.shape[0]))
        Hpauli_rep = np.tensordot(_prep(self.nq),self.H_fci,axes = 1).trace(axis1 = -1,axis2 = -2)
        coords = np.argwhere(Hpauli_rep)
        self.H_qiskit = SparsePauliOp(num_to_let(coords.tolist()),[Hpauli_rep[tuple(c)] for c in coords])
        op = operators(qedfci_obj.mol)
        self.a_ij = op.a_ij
        self.a_ijkl = op.a_ijkl

        self.b_op = np.sqrt(np.vstack([np.diag(np.arange(bosnum+1))[1:],np.zeros(bosnum+1)]))
        self.b_op+=self.b_op.T
        temp = np.zeros((3,2,2,2,2,8,8))
        for sp in range(3):
            for i, j, k, l in product(range(2), repeat = 4):
                temp[sp,i,j,k,l] = np.kron(np.eye(2),op.a_ijkl[sp,i,j,k,l]).todense()
        self.rdm2map = sparse.COO.from_numpy(np.tensordot(temp,_prep(self.nq),axes = ((-1,-2),(-1,-2)))).round(8)
        
        temp = np.zeros((2,2,2,8,8))
        for sp, i, j in product(range(2),repeat = 3):
            temp[sp,i,j] = np.matmul(np.kron(self.b_op,op.a_ij[sp,i,j]),self.H_fci)-np.matmul(self.H_fci,np.kron(self.b_op,op.a_ij[sp,i,j]))
        self.mixresmap = sparse.COO.from_numpy(np.tensordot(temp,_prep(self.nq),axes = ((-1,-2),(-1,-2)))).round(8)

        temp = np.zeros((2,2,2,8,8))
        for sp, i, j in product(range(2),repeat = 3):
            temp[sp,i,j] = np.kron(np.eye(2),op.a_ij[sp,i,j]).todense()
        self.rdm1map = sparse.COO.from_numpy(np.tensordot(temp,_prep(self.nq),axes = ((-1,-2),(-1,-2)))).round(8)

        b = np.sqrt(np.vstack([np.diag(np.arange(bosnum+1))[1:],np.zeros(bosnum+1)]))
        temp = np.kron(np.matmul(b.T,b),np.eye(4))
        self.bosmap = sparse.COO.from_numpy(np.tensordot(temp,_prep(self.nq),axes = ((-1,-2),(-1,-2)))).round(8)


        self.hammap = np.tensordot(_prep(self.nq),self.H_fci,axes = 2).round(8)
        
        self.tuple_out_rdm2 = graph_red((np.argwhere(self.rdm2map)[:,5:]).tolist())

        self.tuple_out_mix = graph_red(np.argwhere(self.mixresmap)[:,3:].tolist())

        self.tuple_out_ham = graph_red(np.argwhere(self.hammap).tolist())

        self.tuple_out_rdm1 = graph_red(np.argwhere(self.rdm1map)[:,3:].tolist())

        self.tuple_out_bos = graph_red(np.argwhere(self.bosmap).tolist())

        self.shots = 8192
        self.delta = .01
        self.sim = AerSimulator()
        self.delta = .01

        self.num_qf = int(np.log2(self.H_fci.shape[0]-bosnum-1))
        self.num_qb = int(np.log2(bosnum+1))


    def resops(self,ket = None, qc = None):
        funcdic = {
                'mix1A': self.mix1A,
                'ferm2A': self.ferm2A,
                }
        ress = []
        ops = []
        for i, res in enumerate(self.resstr):
            tres,top = funcdic[res](ket, qc)
            ress.append(tres)
            try:
                ops.append(top.todense())
            except AttributeError:
                ops.append(top)
        return ress, ops

    def measure(self,qcs,tuple_out,mapping):
        estimator = Estimator(backend = self.sim,options={"resilience_level": 2,"default_shots":self.shots})
        observables = [SparsePauliOp(num_to_let(list(meas))) for (meas) in tuple_out]
        pm = generate_preset_pass_manager(optimization_level = 2, backend = self.sim)
        vec = np.zeros(tuple([len(qcs)])+tuple([4 for i in range(self.nq)]))
        isa_circuit = pm.run(qcs)
        isa_observables = [ob.apply_layout(isa_circuit[0].layout) for ob in observables]
        job = estimator.run([(isa_c,isa_observables) for isa_c in isa_circuit])
        res = job.result()
        for k in range(len(qcs)):
            for ind1, (key,) in enumerate(tuple_out):
                vec[k][key] = res[k].data.evs[ind1]
        return np.tensordot(vec,mapping,axes = ((-1,-2,-3),(-1,-2,-3)))


    def mix1A(self,ket = None, qc = None):
        nq = self.num_qb+self.num_qf
        if qc is not None:
            pass
        elif ket is not None:
            nq = self.num_qb+self.num_qf
            qc = QuantumCircuit(nq,nq)
            qc.initialize(ket)
        else:
            print("need either ket or quantum circuit")
            exit()
        res = self.measure([qc], self.tuple_out_mix, self.mixresmap)

        return res, np.kron(self.b_op,np.tensordot(res[0],self.a_ij,axes = 3))

    def ferm2A(self,ket = None, qc = None):
        nq = self.num_qb+self.num_qf
        if qc is not None:
            qcf = qc.copy()
            qcb = qc.copy()
        elif ket is not None:
            qc = QuantumCircuit(nq,nq)
            qc.initialize(ket)
            qcf = qc.copy()
            qcb = qc.copy()
        else:
            print("need either ket or quantum circuit")
            exit()
        qcf.append(PauliEvolutionGate(self.H_qiskit,time = self.delta),range(self.nq))
        qcb.append(PauliEvolutionGate(self.H_qiskit,time = -self.delta),range(self.nq))
        res = self.measure([qcf,qcb], self.tuple_out_rdm2, self.rdm2map)

        res = res[0]-res[1]
        op = np.tensordot(res[0],self.a_ijkl[0]+self.a_ijkl[2],axes = ((0,1,2,3),(0,1,2,3)))
        op += np.tensordot(res[1],self.a_ijkl[1],axes = ((0,1,2,3),(0,1,2,3)))
        op += np.tensordot(res[1].swapaxes(0,1),self.a_ijkl[1].swapaxes(0,1),axes = ((0,1,2,3),(0,1,2,3)))
        op += np.tensordot(res[1].swapaxes(2,3),self.a_ijkl[1].swapaxes(2,3),axes = ((0,1,2,3),(0,1,2,3)))
        op += np.tensordot(res[1].swapaxes(0,1).swapaxes(2,3),self.a_ijkl[1].swapaxes(0,1).swapaxes(2,3),axes = ((0,1,2,3),(0,1,2,3)))
        op = op/2/self.delta/1.j
        op = np.kron(np.eye(2**self.num_qb),op) 
        return res, op

    def energy(self,ket = None,qc = None):
        if qc is not None:
            pass
        elif ket is not None:
            nq = self.num_qb+self.num_qf
            qc = QuantumCircuit(nq,nq)
            qc.initialize(ket)
            print("bad")
        else:
            print("need either ket or quantum circuit")
            exit()
        en = self.measure([qc],self.tuple_out_ham,self.hammap)
        return en

    def meas_rdm(self,ket = None,qc = None):
        if qc is not None:
            pass
        elif ket is not None:
            nq = self.num_qb+self.num_qf
            qc = QuantumCircuit(nq,nq)
            qc.initialize(ket)
        else:
            print("need either ket or quantum circuit")
            exit()
        qc = [qc]
        estimator = Estimator(backend = self.sim,options={"resilience_level": 2})
        tuple_out = self.tuple_out_rdm1.copy()
        tuple_out.extend(self.tuple_out_bos.copy())
        observables = [SparsePauliOp(num_to_let(list(meas))) for (meas) in tuple_out]
        pm = generate_preset_pass_manager(optimization_level = 3, backend = self.sim)
        vec = np.zeros(tuple([len(qc)])+tuple([4 for i in range(self.nq)]))
        isa_circuit = pm.run(qc)
        isa_observables = [ob.apply_layout(isa_circuit[0].layout) for ob in observables]
        job = estimator.run([(isa_c,isa_observables) for isa_c in isa_circuit])
        res = job.result()
        for k in range(len(qc)):
            for ind1, (key,) in enumerate(self.tuple_out_rdm1):
                vec[k][key] = res[k].data.evs[ind1]
        rdm1 = np.tensordot(vec,self.rdm1map,axes = ((-1,-2,-3),(-1,-2,-3)))
        vec = np.zeros(tuple([len(qc)])+tuple([4 for i in range(self.nq)]))
        for k in range(len(qc)):
            for ind1, (key,) in enumerate(self.tuple_out_bos):
                vec[k][key] = res[k].data.evs[ind1+len(self.tuple_out_rdm1)]
        bos = np.tensordot(vec,self.bosmap,axes = ((-1,-2,-3),(-1,-2,-3)))
        return rdm1, bos

