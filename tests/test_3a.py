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
ngrid = np.ones_like(omega)
######################################################################

######################################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRRORS
######################################################################
omegaPR = 2.0
gammaPR = 0.05

mirror_1 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=False)
mirror_2 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=True)
air_infty = ts.LayerPhysical(n=ngrid,k=0*ngrid,mu=ngrid,d=np.inf)

# #####################################################################

theta0 = 0.231 # INCIDENT ANGLE

l = np.linspace(150, 450, 20)
ampl = list()
for dist in l:
    air_cavity = ts.LayerPhysical(n=ngrid,k=0*ngrid,mu=ngrid,d=dist)
    layers = [air_infty, mirror_1, air_cavity, mirror_2, air_infty]
    tScat = ts.TScat(theta0, layers, omega)

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