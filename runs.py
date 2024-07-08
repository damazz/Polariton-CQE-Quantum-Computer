'''
Importing some basic packages
'''

import numpy as np
import numpy.linalg as LA
from functools import reduce
from pyscf import gto, scf

'''
Importing some custom packages that contain QED code and ACSE objects
'''

from QEDHF import qedhf
from ACSE_quant import ACSE

'''
Now we are naming a simulator object
We have commented out the runtime backend, and are using the AerSimulator
You could also use a fakebackend as indicated
'''

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


'''
Here we are defining the dissociation range for H2
and some lists to store our energy, 1-RDMs, and expected boson populations
'''

ran = np.linspace(.5,2.5,7)
en = []
rdm = []
boson_pop = []


for ri,r in enumerate(ran):
        '''
        Creating a Pyscf Hartree Fock Obj
        '''
        mol = gto.M(atom = f"H 0 0 0; H 0 0 {r}",
                basis = "sto3g",
                unit = "A",
                symmetry = False,)
        mf = scf.HF(mol)
        mf.kernel()
        myhf = scf.HF(mol)
        myhf.kernel()

        '''
        Creating a QED-HF object, need to specify the lambda vector
        '''
        lamb = np.array([0.0,0.0,0.2])
        myqed  = qedhf(myhf,lamb)
        myqed.kernel()

        '''
        Starting the ACSE
        ACSE(QEDHF_obj,boson_excitions,['fermionic_residual','mixed_residual'])
        myacse.delta = the forward and backward time propogation amount for 2-RDM calculation
        myacse.sim = the above specified simulator
        myacse.run(number of steps in the ACSE)
        '''
        myacse = ACSE(myqed,1, ['ferm2A','mix1A'])
        myacse.delta = .5
        myacse.sim = sim
        myacse.run(5)

        '''
        Collecting the results!
        '''
        en.append(np.min(myacse.e))
        rdm.append(myacse.rdm[np.argmin(myacse.e)][0])
        boson_pop.append(myacse.rdm[np.argmin(myacse.e)][1][0])
        print((en[ri]-LA.eigh(myacse.H_fci)[0][0])*1000)


