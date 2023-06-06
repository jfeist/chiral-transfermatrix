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
        k = k0
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

Mmat_1 = ts.chirality_preserving_mirror_transfermatrix(omegaPR,gammaPR,omega,reversed=False)
Mmat_2 = ts.chirality_preserving_mirror_transfermatrix(omegaPR,gammaPR,omega,reversed=True)

coupl = np.linspace(0.0, 1.0, 20)

Tplist = []
Tmlist = []
Rplist = []
Rmlist = []
DCTlist = []
DCAlist = []

for coup in coupl:

    ################
    # INCIDENT ANGLE
    ################
    theta0 = 0
    ################

    ######
    # AIR
    ###############
    n1 = 1 * ngrid
    mu1 = 1 * ngrid
    k1 = 0 * ngrid
    d1 = np.inf
    ###############

    #####################
    # PRESERVING MIRROR 1
    #######################################
    k2 = 0 * ngrid
    mu2 = 1 * ngrid
    n2 = 1 * ngrid
    d2 = 0  # the distance has no influence
    #######################################

    #####
    # AIR
    ###############
    n3 = 1 * ngrid
    mu3 = 1 * ngrid
    k3 = 0 * ngrid
    d3 = 0.01
    ###############

    #################
    # CHIRAL MATERIAL
    #########################################################################################
    epsinf = 2.89
    omegapChiral = coup
    eps4M, n4, k4 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 0.0)
    mu4 = 1 * ngrid
    k4 = 0 * ngrid
    dL = 180
    #########################################################################################

    #####
    # AIR
    ###############
    n5 = 1 * ngrid
    mu5 = 1 * ngrid
    k5 = 0 * ngrid
    d5 = 0.01
    ###############

    #####################
    # PRESERVING MIRROR 2
    ######################################
    k6 = 0 * ngrid
    mu6 = 1 * ngrid
    n6 = 1 * ngrid
    d6 = 0 # the distance has no influence
    ######################################

    #####
    # AIR
    ###############
    n7 = 1 * ngrid
    mu7 = 1 * ngrid
    k7 = 0 * ngrid
    d7 = np.inf
    ###############

    ########################################
    # ALL THE ARRAYS OF THE INPUT PARAMETERS
    #############################################################################
    nTOT = [n1, n2, n3, n4, n5, n6, n7]
    muTOT = [mu1, mu2, mu3, mu4, mu5, mu6, mu7]
    kTOT = [k1, k2, k3, k4, k5, k6, k7]
    dTOT = [d1, d2, d3, dL, d5, d6, d7]
    matTOT = ['air', Mmat_1, 'air', 'ChiralMat', 'air', Mmat_2, 'air']
    #############################################################################

    ###########################################
    # CALLING OF THE CLASS FOR THE EMPTY CAVITY
    #########################################################################
    tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT)
    #########################################################################

    Tplist.append(tScat.Tsp)
    Tmlist.append(tScat.Tsm)
    Rplist.append(tScat.Rsp)
    Rmlist.append(tScat.Rsm)
    DCTlist.append(tScat.dct_s)

#############
# OBSERVABLES
#######################
arr1 = np.array(Tplist)
arr2 = np.array(Tmlist)
arr3 = np.array(Rplist)
arr4 = np.array(Rmlist)
arr5 = np.array(DCTlist)
#######################

# np.savez_compressed('tests/test_2.npz', omega=omega, coupl=coupl, Tplist=arr1, Tmlist=arr2, Rplist=arr3, Rmlist=arr4, DCTlist=arr5)
def test_2():    
    ref_data = np.load('tests/test_2.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(coupl, ref_data['coupl'])
    assert np.allclose(arr1, ref_data['Tplist'])
    assert np.allclose(arr2, ref_data['Tmlist'])
    assert np.allclose(arr3, ref_data['Rplist'])
    assert np.allclose(arr4, ref_data['Rmlist'])
    assert np.allclose(arr5, ref_data['DCTlist'])

    print('Test 2 passed!')