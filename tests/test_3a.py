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
######################################################################

def chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega,reversed=False):
    ngrid = np.ones_like(omega)
    tP =  gammaPR / (1j * (omega - omegaPR) + gammaPR)
    rM = abs(tP)
    phase = tP / rM
    tPM = abs(tP)
    t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
    phit = np.pi / 2
    pst = np.exp(1j * phit)

    tPP_r = t * pst
    tMM_r = t * pst
    tPP_l = t * pst
    tMM_l = t * pst
    if reversed:
        tMP_r = tPM * phase
        tPM_r = 0.0j * ngrid
        tMP_l = 0.0j * ngrid
        tPM_l = tPM * phase
    else:
        tMP_r = 0.0j * ngrid
        tPM_r = tPM * phase
        tMP_l = tPM * phase
        tPM_l = 0.0j * ngrid

    if reversed:
        rPP_r = 0.0j * ngrid
        rMM_r = tPM * pst**4 / phase**3
        rPP_l = - tPM * phase
        rMM_l = 0.0j * ngrid
    else:
        rPP_r = tPM * pst**4 / phase**3
        rMM_r = 0.0j * ngrid
        rPP_l = 0.0j * ngrid
        rMM_l = - tPM * phase
    rMP_r = - t / phase**2 * pst**3
    rPM_r = - t / phase**2 * pst**3
    rMP_l = t * phase**2 / pst
    rPM_l = t * phase**2 / pst

    # 2x2 scattering matrices
    t_right = [tPP_r, tMP_r, tPM_r, tMM_r]
    t_left  = [tPP_l, tMP_l, tPM_l, tMM_l]
    r_right = [rPP_r, rMP_r, rPM_r, rMM_r]
    r_left  = [rPP_l, rMP_l, rPM_l, rMM_l]

    return [t_right, t_left, r_right, r_left]

######################################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRRORS
######################################################################
omegaPR = 2.0
gammaPR = 0.05

scatTOT = [chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega,reversed=False),
           chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega,reversed=True)]

# #####################################################################

theta0 = 0.231 # INCIDENT ANGLE

ngrid = np.ones_like(omega)

l = np.linspace(150, 450, 20)
ampl = list()
for dist in l:
    # five layers: air, chirality-preserving mirror, air, reversed
    # chirality-preserving mirror, air note that since the chirality-preserving
    # mirrors are handled by directly passing the scattering matrices, the parameters d,n,mu,k do not matter for them.

    mats = [  'air', 'Custom',   'air', 'Custom',   'air']
    ds   = [ np.inf,       0.,    dist,       0.,  np.inf]
    ns   = [  ngrid,    ngrid,   ngrid,    ngrid,   ngrid]
    mus  = [  ngrid,    ngrid,   ngrid,    ngrid,   ngrid]
    ks   = [0*ngrid,  0*ngrid, 0*ngrid,  0*ngrid, 0*ngrid]

    tScat = ts.TScat(theta0, ns, mus, ks, ds, omega, mats, scatTOT)

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

# np.savez_compressed('tests/test_3a.npz', omega=omega, ELp=Elp, ELm=Elm, ERp=Erp, ERm=Erm, lcp=lcp, rcp=rcp)
def test_3a():
    ref_data = np.load('tests/test_3a.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(Elp, ref_data['ELp'])
    assert np.allclose(Elm, ref_data['ELm'])
    assert np.allclose(Erp, ref_data['ERp'])
    assert np.allclose(Erm, ref_data['ERm'])
    assert np.allclose(lcp, ref_data['lcp'])
    assert np.allclose(rcp, ref_data['rcp'])

    print('Test 3a passed!')