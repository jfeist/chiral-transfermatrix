import tscat as ts
import numpy as np

def eps_DL(omega, epsinf, omegap, omega0=0, gamma=0, k0=0):
    """Drude-Lorentz model for the dielectric function of a material."""
    eps = epsinf + omegap**2 / (omega0**2 - omega**2 - 1j * gamma * omega)
    k = 0*eps if k0==0 else k0 * omega / omega0 * (eps-epsinf)
    return eps, k

omega = np.linspace(1.6, 2.4, 30)
theta0 = 0.231

omegaPR = 2.0
gammaPR = 0.05

mirror_1 = ts.chirality_preserving_mirror(omega,omegaPR,gammaPR,reversed=False)
mirror_2 = ts.chirality_preserving_mirror(omega,omegaPR,gammaPR,reversed=True)
air_infty = ts.MaterialLayer(d=np.inf, eps=1)
air_thin  = ts.MaterialLayer(d=0.01,   eps=1)

# we want to scan over the resonance strength omegapChiral AND the frequency
# omega (i.e., output should be 100x100), so make this a 100x1 2D array
# numpy broadcasting will take care of the rest (i.e., omega will be the second axis)
omegapChiral = np.linspace(0.0, 1.0, 20)[:,None]
eps_mol, k_mol = eps_DL(omega, epsinf=2.89, omegap=omegapChiral, omega0=2.0, gamma=0.05, k0=0.0)
molecules = ts.MaterialLayer(d=180.,eps=eps_mol,kappa=k_mol)

layers = [air_infty, mirror_1, air_thin, molecules, air_thin, mirror_2, air_infty]

tScat = ts.TScat(layers, omega, theta0)

# np.savez_compressed('tests/test_2a.npz', omega=omega, coupl=coupl, Tplist=arr1, Tmlist=arr2, Rplist=arr3, Rmlist=arr4, DCTlist=arr5)
def test_2a():
    ref_data = np.load('tests/test_2a.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(omegapChiral.squeeze(), ref_data['coupl'])
    assert np.allclose(tScat.Tsp, ref_data['Tplist'])
    assert np.allclose(tScat.Tsm, ref_data['Tmlist'])
    assert np.allclose(tScat.Rsp, ref_data['Rplist'])
    assert np.allclose(tScat.Rsm, ref_data['Rmlist'])
    assert np.allclose(tScat.DCTs, ref_data['DCTlist'])

    print('Test 2a passed!')