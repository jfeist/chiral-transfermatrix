import tscat as ts
import numpy as np

# we want to scan over frequency omega, and cavity length l, as two different
# axis. TScat is "transparent" to numpy broadcasting, so we can just make omega
# a 1D array and l a 2D array (shape nl x 1), and numpy will take care of the
# rest. The results will be 2D arrays of shape nl x nomega
omega = np.linspace(1.8, 2.2, 30)
l = np.linspace(150, 450, 20)[:,None]
omegaPR = 2.0
gammaPR = 0.05
mirror_1 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=False)
mirror_2 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=True)
air_infty = ts.MaterialLayer(n=1,k=0,mu=1,d=np.inf)
air_cavity = ts.MaterialLayer(n=1,k=0,mu=1,d=l)
layers = [air_infty, mirror_1, air_cavity, mirror_2, air_infty]

theta0 = 0 # INCIDENT ANGLE

tScat = ts.TScat(theta0, layers, omega)

ampl = tScat.field_ampl(2, [1,0])  # field in cavity for an incoming LCP wave

Elp = ampl[:, :, 0]
Elm = ampl[:, :, 1]
Erp = ampl[:, :, 2]
Erm = ampl[:, :, 3]
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

    print('Test 3 passed!')