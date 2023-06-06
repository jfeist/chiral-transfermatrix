###########
# LIBRARIES
######################################################
import tscat as ts  # Essential "import" to run TSCAT

import numpy as np
######################################################


###################################################################
# GENERAL DEFINITION OF THE DIELECTRIC FUNCTION AND CHIRAL COUPLING
################################################################################################
def eps_DL(epsinf, omegap, omega, omega0=0, gamma=0, k0=0):
    eps = epsinf + (omegap**2 / ((omega0**2 - omega**2) - 1j * gamma * omega))
    # dispersive dielectric function
    n = np.sqrt(eps)

    if k0 != 0: # chiral coupling
        k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))
    else:
        k = np.zeros_like(omega)
    return eps, n, k
#################################################################################################

######################################################################
# RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
######################################################################
omega = np.linspace(1.6, 2.4, 30)
ngrid = np.ones_like(omega)
######################################################################

###############################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 1
######################################################################
omegaPR = 2.0
gammaPR = 0.05

mirror_1 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=False)
mirror_2 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=True)
air_infty = ts.LayerPhysical(n=ngrid,k=0*ngrid,mu=ngrid,d=np.inf)
air_thin  = ts.LayerPhysical(n=ngrid,k=0*ngrid,mu=ngrid,d=0.01)

coupl = np.linspace(0.0, 1.0, 20)

tScats = []
for coup in coupl:

    ################
    # INCIDENT ANGLE
    ################
    theta0 = 0.231
    ################

    #################
    # CHIRAL MATERIAL
    #########################################################################################
    epsinf = 2.89
    omegapChiral = coup
    eps4M, n4, k4 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 0.0)
    molecules = ts.LayerPhysical(n=n4,k=k4,mu=ngrid,d=180.)
    #########################################################################################

    layers = [air_infty, mirror_1, air_thin, molecules, air_thin, mirror_2, air_infty]

    ###########################################
    # CALLING OF THE CLASS FOR THE EMPTY CAVITY
    #########################################################################
    tScats.append(ts.TScat(theta0, layers, omega))
    #########################################################################

#############
# OBSERVABLES
#######################
arr1 = np.array([ts.Tsp for ts in tScats])
arr2 = np.array([ts.Tsm for ts in tScats])
arr3 = np.array([ts.Rsp for ts in tScats])
arr4 = np.array([ts.Rsm for ts in tScats])
arr5 = np.array([ts.dct_s for ts in tScats])
#######################

# np.savez_compressed('tests/test_2a.npz', omega=omega, coupl=coupl, Tplist=arr1, Tmlist=arr2, Rplist=arr3, Rmlist=arr4, DCTlist=arr5)
def test_2a():
    ref_data = np.load('tests/test_2a.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(coupl, ref_data['coupl'])
    assert np.allclose(arr1, ref_data['Tplist'])
    assert np.allclose(arr2, ref_data['Tmlist'])
    assert np.allclose(arr3, ref_data['Rplist'])
    assert np.allclose(arr4, ref_data['Rmlist'])
    assert np.allclose(arr5, ref_data['DCTlist'])

    print('Test 2a passed!')