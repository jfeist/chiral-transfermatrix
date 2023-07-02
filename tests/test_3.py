import tscat as ts
import numpy as np

def test_3():
    # we want to scan over the angle theta0, cavity length l, and frequency omega
    # (i.e., output should be 2x600x500),
    # so make the arrays to get this with broadcasting - index order is then
    # itheta0, iomegap, iomega
    omega = np.linspace(1.8, 2.2, 30)
    l = np.linspace(150, 450, 20)[:,None]
    theta0 = np.r_[0, 0.231][:,None,None]

    omegaPR = 2.0
    gammaPR = 0.05
    mirror_1 = ts.helicity_preserving_mirror(omega,omegaPR,gammaPR,reversed=False)
    mirror_2 = ts.helicity_preserving_mirror(omega,omegaPR,gammaPR,reversed=True)
    air_infty  = ts.MaterialLayer(d=np.inf,eps=1)
    air_cavity = ts.MaterialLayer(d=l,     eps=1)
    layers = [air_infty, mirror_1, air_cavity, mirror_2, air_infty]

    tScat = ts.TScat(layers, omega, theta0)

    ampl = tScat.field_ampl(2, [1,0])  # field in cavity for an incoming LCP wave

    # np.savez_compressed("tests/test_3.npz", ampl=ampl)
    ref_data = np.load('tests/test_3.npz')
    assert np.allclose(ref_data["ampl"], ampl)
