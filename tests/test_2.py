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

    if k0 != 0:
        k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))
        # chiral coupling
        return eps, n, k

    else:
        return eps, n, k0
#################################################################################################


######################################################################
# RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
######################################################################
omega = np.linspace(1.6, 2.4, 30)
ngrid = np.ones_like((omega))
######################################################################

scatTOT = list()

###############################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 1
######################################################################
omegaPR = 2.0
gammaPR = 0.05

tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
rM =  np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
phase = tP / rM
tPM=np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
phit = np.pi / 2
pst = np.exp(1j * phit)

tPP_r = t * pst
tMP_r = 0.0j * ngrid
tPM_r = tPM * phase
tMM_r = t * pst

tPP_l = t * pst
tMP_l = tPM * phase
tPM_l = 0.0j * ngrid
tMM_l = t * pst

rPP_r = tPM * pst**4 * (1 / phase)**3
rMP_r = - t * (1 / phase)**2 * (pst**3)
rPM_r = - t * (1 / phase)**2 * (pst**3)
rMM_r = 0.0j * ngrid

rPP_l = 0.0j * ngrid
rMP_l = t * (phase**2) * (1 / pst)
rPM_l = t * (phase**2) * (1 / pst)
rMM_l = - tPM * phase

t1_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
t1_left = [tPP_l, tMP_l, tPM_l, tMM_l]
r1_right = [rPP_r, rMP_r, rPM_r, rMM_r]
r1_left = [rPP_l, rMP_l, rPM_l, rMM_l]

scatTOT.append([t1_right, t1_left, r1_right, r1_left])
#####################################################################

###############################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 2
######################################################################
omegaPR = 2.0
gammaPR = 0.05

tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
rM  = np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
phase = tP / rM
tPM = np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
phit = np.pi / 2
pst = np.exp(1j * phit)

tPP_r = t * pst
tMP_r = tPM * phase
tPM_r = 0.0j * ngrid
tMM_r = t * pst

tPP_l = t * pst
tMP_l = 0.0j * ngrid
tPM_l = tPM * phase
tMM_l = t * pst

rPP_r = 0.0j * ngrid
rMP_r = - t * (1 / phase)**2 * (pst**3)
rPM_r = - t * (1 / phase)**2 * (pst**3)
rMM_r = tPM * pst**4 * (1 / phase)**3

rPP_l = - tPM * phase
rMP_l = t * (phase**2) * (1 / pst)
rPM_l = t * (phase**2) * (1 / pst)
rMM_l = 0.0j * ngrid

t2_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
t2_left = [tPP_l, tMP_l, tPM_l, tMM_l]
r2_right = [rPP_r, rMP_r, rPM_r, rMM_r]
r2_left = [rPP_l, rMP_l, rPM_l, rMM_l]

scatTOT.append([t2_right,t2_left,r2_right,r2_left])
###################################################################

coupl = np.linspace(0.0, 1.0, 20)

Tplist = []
Tmlist = []
Rplist = []
Rmlist = []
DCTlist = []
DCAlist = []

for i in range(len(coupl)):

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
    omegapChiral = coupl[i]
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
    matTOT = ['air', scatTOT[0], 'air', 'ChiralMat', 'air', scatTOT[1], 'air']
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