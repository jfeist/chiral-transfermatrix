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
    
    if k0 != 0:
        k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))  
        # chiral coupling
        return eps, n, k

    else:
        return eps, n, k0
###############################################################################################


######################################################################
# RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
######################################################################
omega = np.linspace(1.6, 2.4, 30)  # Omega in eV
ngrid = np.ones_like((omega))
######################################################################

coupl = np.linspace(0.0, 1.0, 20)

Tplist = []
Tmlist = []
Rplist = []
Rmlist = []
DCTlist = []

for i in range(len(coupl)):
 
    ################
    # INCIDENT ANGLE
    ################
    theta0 = 0
    ################
    
    #####
    # AIR
    ###############
    n1 = 1 * ngrid
    mu1 = 1 * ngrid
    k1 = 0 * ngrid
    d1 = np.inf
    ###############
    
    ########
    # MIRROR
    ########################################################################################
    epsinf = 4.77574276
    omegapMirr = 9.48300763
    eps2, n2, k2 = eps_DL(epsinf, omegapMirr, omega, omega0 = 0, gamma = 0.17486845, k0 = 0)
    mu2 = 1 * ngrid
    k2 = 0 * ngrid
    d2 = 30
    ########################################################################################   

    #################
    # CHIRAL MATERIAL
    ##########################################################################################
    epsinf = 2.89
    omegapChiral = coupl[i]
    eps3M, n3, k3 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 1e-3)
    mu3 = 1 * ngrid
    dL = 150
    ########################################################################################## 

    ########
    # MIRROR
    ########################################################################################
    epsinf = 4.77574276
    omegapMirr = 9.48300763
    eps4, n4, k4 = eps_DL(epsinf, omegapMirr, omega, omega0 = 0, gamma = 0.17486845, k0 = 0)
    mu4 = 1 * ngrid
    k4 = 0 * ngrid
    d4 = 30
    ######################################################################################## 

    #####
    # AIR
    ###############
    n5 = 1 * ngrid
    mu5 = 1 * ngrid
    k5 = 0 * ngrid
    d5 = np.inf
    ###############
    
    ########################################
    # ALL THE ARRAYS OF THE INPUT PARAMETERS
    ####################################################
    nTOT = [n1, n2, n3, n4, n5]
    muTOT = [mu1, mu2, mu3, mu4, mu5]
    kTOT = [k1, k2, k3, k4, k5] 
    dTOT = [d1, d2, dL, d4, d5] 
    matTOT = ['air', 'mirr', 'ChiralMat', 'mirr', 'air']
    ####################################################
    
    ######################
    # CALLING OF THE CLASS
    ################################################################
    tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT)  
    ################################################################
    
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

# np.savez_compressed('tests/test_1.npz', omega=omega, coupl=coupl, Tplist=arr1, Tmlist=arr2, Rplist=arr3, Rmlist=arr4, DCTlist=arr5)
def test_1():
    ref_data = np.load('tests/test_1.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(coupl, ref_data['coupl'])
    assert np.allclose(arr1, ref_data['Tplist'])
    assert np.allclose(arr2, ref_data['Tmlist'])
    assert np.allclose(arr3, ref_data['Rplist'])
    assert np.allclose(arr4, ref_data['Rmlist'])
    assert np.allclose(arr5, ref_data['DCTlist'])

    print('Test 1 passed!')