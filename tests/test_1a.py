###########
# LIBRARIES
#####################################################
import tscat as ts  # Essential "import" to run TSCAT

import numpy as np
#####################################################


###################################################################
# GENERAL DEFINITION OF THE DIELECTRIC FUNCTION AND CHIRAL COUPLING
###############################################################################################
def eps_DL(epsinf, omegap, omega, omega0=0, gamma=0, k0=0):
    eps = epsinf + (omegap**2 / ((omega0**2 - omega**2) - 1j * gamma * omega))
    # dispersive dielectric function
    n = np.sqrt(eps)

    if k0 != 0: # chiral coupling
        k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))
    else:
        k = np.zeros_like(omega)
    return eps, n, k
###############################################################################################


######################################################################
# RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
######################################################################
omega = np.linspace(1.6, 2.4, 30)  # Omega in eV
ngrid = np.ones_like(omega)
######################################################################

################
# INCIDENT ANGLE
################
theta0 = 0.231
################

epsinf = 4.77574276
omegapMirr = 9.48300763
eps_m, n_m, k_m = eps_DL(epsinf, omegapMirr, omega, omega0 = 0, gamma = 0.17486845, k0 = 0)
metal_mirror = ts.MaterialLayer(n=n_m,k=k_m,mu=ngrid,d=30)

air_infty = ts.MaterialLayer(n=ngrid,k=0*ngrid,mu=ngrid,d=np.inf)

tScats = []
omegapChirals = np.linspace(0.0, 1.0, 20)
for omegapChiral in omegapChirals:
    # CHIRAL MATERIAL
    epsinf = 2.89
    eps3M, n3, k3 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 1e-3)
    molecules = ts.MaterialLayer(n=n3,k=k3,mu=ngrid,d=150.)

    layers = [air_infty, metal_mirror, molecules, metal_mirror, air_infty]
    tScats.append(ts.TScat(theta0, layers, omega))

#############
# OBSERVABLES
#######################
arr1 = np.array([ts.Tsp for ts in tScats])
arr2 = np.array([ts.Tsm for ts in tScats])
arr3 = np.array([ts.Rsp for ts in tScats])
arr4 = np.array([ts.Rsm for ts in tScats])
arr5 = np.array([ts.dct_s for ts in tScats])
#######################

# np.savez_compressed('tests/test_1a.npz', omega=omega, coupl=coupl, Tplist=arr1, Tmlist=arr2, Rplist=arr3, Rmlist=arr4, DCTlist=arr5)
def test_1a():
    ref_data = np.load('tests/test_1a.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(omegapChirals, ref_data['coupl'])
    assert np.allclose(arr1, ref_data['Tplist'])
    assert np.allclose(arr2, ref_data['Tmlist'])
    assert np.allclose(arr3, ref_data['Rplist'])
    assert np.allclose(arr4, ref_data['Rmlist'])
    assert np.allclose(arr5, ref_data['DCTlist'])

    print('Test 1a passed!')