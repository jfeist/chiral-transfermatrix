import tscat as ts
import numpy as np

def eps_DL(omega, epsinf, omegap, omega0=0, gamma=0, k0=0):
    """Drude-Lorentz model for the dielectric function of a material."""
    res = omegap**2 / (omega0**2 - omega**2 - 1j * gamma * omega)
    eps = epsinf + res
    n = np.sqrt(eps)
    k = 0*eps if k0==0 else k0 * omega / omega0 * res
    return eps, n, k

omega = np.linspace(1.6, 2.4, 30)
theta0 = 0.231

omegaPR = 2.0
gammaPR = 0.05

mirror_1 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=False)
mirror_2 = ts.chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=True)
air_infty = ts.MaterialLayer(n=1,k=0,mu=1,d=np.inf)
air_thin  = ts.MaterialLayer(n=1,k=0,mu=1,d=0.01)

# we want to scan over the resonance strength omegapChiral AND the frequency
# omega (i.e., output should be 100x100), so make this a 100x1 2D array
# numpy broadcasting will take care of the rest (i.e., omega will be the second axis)
omegapChiral = np.linspace(0.0, 1.0, 20)[:,None]
eps_mol, n_mol, k_mol = eps_DL(omega, epsinf=2.89, omegap=omegapChiral, omega0=2.0, gamma=0.05, k0=0.0)
molecules = ts.MaterialLayer(n=n_mol,k=k_mol,mu=1,d=180.)

layers = [air_infty, mirror_1, air_thin, molecules, air_thin, mirror_2, air_infty]

tScat = ts.TScat(theta0, layers, omega)

# np.savez_compressed('tests/test_2a.npz', omega=omega, coupl=coupl, Tplist=arr1, Tmlist=arr2, Rplist=arr3, Rmlist=arr4, DCTlist=arr5)
def test_2a():
    ref_data = np.load('tests/test_2a.npz')
    assert np.allclose(omega, ref_data['omega'])
    assert np.allclose(omegapChiral.squeeze(), ref_data['coupl'])
    assert np.allclose(tScat.Tsp, ref_data['Tplist'])
    assert np.allclose(tScat.Tsm, ref_data['Tmlist'])
    assert np.allclose(tScat.Rsp, ref_data['Rplist'])
    assert np.allclose(tScat.Rsm, ref_data['Rmlist'])
    assert np.allclose(tScat.dct_s, ref_data['DCTlist'])

    print('Test 2a passed!')