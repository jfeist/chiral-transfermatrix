###########
# LIBRARIES
######################################################
import tscat as ts  # Essential "import" to run TSCAT

import numpy as np
######################################################

######################################################################
# RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
######################################################################
omega = np.linspace(1.8, 2.2, 30)
######################################################################

######################################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRRORS
######################################################################
omegaPR = 2.0
gammaPR = 0.05

Mmat_1 = ts.chirality_preserving_mirror_transfermatrix(omegaPR,gammaPR,omega,reversed=False)
Mmat_2 = ts.chirality_preserving_mirror_transfermatrix(omegaPR,gammaPR,omega,reversed=True)

# #####################################################################

theta0 = 0.231 # INCIDENT ANGLE

ngrid = np.ones_like(omega)

l = np.linspace(150, 450, 20)
ampl = list()
for dist in l:
    # five layers:
    # 1) air
    # 2) chirality-preserving mirror
    # 3) air
    # 4) reversed chirality-preserving mirror
    # 5) air
    # note that since the chirality-preserving mirrors are handled by directly
    # passing the scattering matrices, the parameters d,n,mu,k do not matter for
    # them.
    mats = [  'air',  Mmat_1,   'air',  Mmat_2,   'air']
    ds   = [ np.inf,      0.,    dist,      0.,  np.inf]
    ns   = [  ngrid,   ngrid,   ngrid,   ngrid,   ngrid]
    mus  = [  ngrid,   ngrid,   ngrid,   ngrid,   ngrid]
    ks   = [0*ngrid, 0*ngrid, 0*ngrid, 0*ngrid, 0*ngrid]

    tScat = ts.TScat(theta0, ns, mus, ks, ds, omega, mats)

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