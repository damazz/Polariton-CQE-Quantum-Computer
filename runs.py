import numpy as np
import numpy.linalg as LA
from functools import reduce
from pyscf import gto, scf
from QEDHF import qedhf

from ACSE_quant import ACSE

#from qiskit_ibm_runtime import QiskitRuntimeService
# service = QiskitRuntimeService(
#     channel='ibm_quantum',
#     instance='',
#     token='',
# )
#sim = service.get_backend(name="")

from qiskit_aer import AerSimulator
sim = AerSimulator()
#from qiskit_ibm_runtime.fake_provider import FakeLagosV2
#sim = FakeLagosV2()


ran = np.linspace(.5,2.5,7)
en = []
rdm = []
boson_pop = []


for ri,r in enumerate(ran):
        mol = gto.M(atom = f"H 0 0 0; H 0 0 {r}",
                basis = "sto3g",
                unit = "A",
                symmetry = False,)
        mf = scf.HF(mol)
        mf.kernel()
        myhf = scf.HF(mol)
        myhf.kernel()
        lamb = np.array([0.0,0.0,0.2])
        myqed  = qedhf(myhf,lamb)
        myqed.kernel()
        myacse = ACSE(myqed,1, ['ferm2A','mix1A'])
        myacse.delta = .5
        myacse.sim = sim
        myacse.run(5)
        en.append(np.min(myacse.e))
        rdm.append(myacse.rdm[np.argmin(myacse.e)][0])
        boson_pop.append(myacse.rdm[np.argmin(myacse.e)][1][0])
        print((en[ri]-LA.eigh(myacse.H_fci)[0][0])*1000)


