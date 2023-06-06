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
omega = np.linspace(1.8, 2.2, 30)
ngrid = np.ones_like(omega)
######################################################################

def chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega):
    tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
    rM = abs(tP)
    phase = tP / rM
    tPM = abs(tP)
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

    rPP_r = tPM * pst**4 / phase**3
    rMP_r = - t / phase**2 * pst**3
    rPM_r = - t / phase**2 * pst**3
    rMM_r = 0.0j * ngrid

    rPP_l = 0.0j * ngrid
    rMP_l = t * phase**2 / pst
    rPM_l = t * phase**2 / pst
    rMM_l = - tPM * phase

    t1_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
    t1_left = [tPP_l, tMP_l, tPM_l, tMM_l]
    r1_right = [rPP_r, rMP_r, rPM_r, rMM_r]
    r1_left = [rPP_l, rMP_l, rPM_l, rMM_l]

    return [t1_right, t1_left, r1_right, r1_left]

###############################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 1
######################################################################
omegaPR = 2.0
gammaPR = 0.05

scatTOT = [chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega)]
#           chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega)]

# #####################################################################

# ###############################################################
# # DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 2
# ######################################################################
# omegaPR = 2.0
# gammaPR = 0.05

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

l = np.linspace(150, 450, 20)
ampl = list()

for dist in l:

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
    d3 = dist
    ###############

    #####################
    # PRESERVING MIRROR 2
    ######################################
    k4 = 0 * ngrid
    mu4 = 1 * ngrid
    n4 = 1 * ngrid
    d4 = 0 # the distance has no influence
    ######################################

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
    ##################################################
    nTOT = [n1, n2, n3, n4, n5]
    muTOT = [mu1, mu2, mu3, mu4, mu5]
    kTOT = [k1, k2, k3, k4, k5]
    dTOT = [d1, d2, d3, d4, d5]
    matTOT = ['air', 'Custom', 'air', 'Custom', 'air']
    ##################################################

    ###########################################
    # CALLING OF THE CLASS FOR THE EMPTY CAVITY
    #########################################################################
    tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT, scatTOT)
    #########################################################################

    ampl.append(tScat.calc_ampl(2, [1,0], omega))  # field in cavity for an incoming LCP wave


#############
# OBSERVABLES
#####################################################################################
ampl2 = np.array(ampl) #.reshape(len(l), len(omega), 4)
Elp = ampl2[:, :, 0]
Elm = ampl2[:, :, 1]
Erp = ampl2[:, :, 2]
Erm = ampl2[:, :, 3]
lcp = abs(Elp)**2 + abs(Erp)**2  # total LCP in layer 2 (inside the cavity)
rcp = abs(Elm)**2 + abs(Erm)**2  # total RCP in layer 2 (inside the cavity)
#####################################################################################

# np.savez_compressed('tests/test_3.npz', omega=omega, ELp=Elp, ELm=Elm, ERp=Erp, ERm=Erm, lcp=lcp, rcp=rcp)
def test_3():
    ref_data = np.load('tests/test_3.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(Elp, ref_data['ELp'])
    assert np.allclose(Elm, ref_data['ELm'])
    assert np.allclose(Erp, ref_data['ERp'])
    assert np.allclose(Erm, ref_data['ERm'])
    assert np.allclose(lcp, ref_data['lcp'])
    assert np.allclose(rcp, ref_data['rcp'])

    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(Elp, ref_data['ELp'],atol=1e-4)
    assert np.allclose(Elm, ref_data['ELm'])
    assert np.allclose(Erp, ref_data['ERp'],atol=1e-4)
    assert np.allclose(Erm, ref_data['ERm'],atol=1e-6)
    assert np.allclose(lcp, ref_data['lcp'],atol=1e-5)
    assert np.allclose(rcp, ref_data['rcp'])

    print('Test 3 passed!')