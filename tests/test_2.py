import transfermat_scatt as ts
import numpy as np

def eps_DL(omega, epsinf, omegap, omega0=0, gamma=0, kappa0=0):
    """Drude-Lorentz model for the dielectric function of a material."""
    assert not (kappa0 != 0 and omega0 == 0), "kappa0 != 0 requires omega0 != 0"
    eps = epsinf + omegap**2 / (omega0**2 - omega**2 - 1j * gamma * omega)
    k = 0*eps if kappa0==0 else kappa0 * omega / omega0 * (eps-epsinf)
    return eps, k

def test_2():
    # we want to scan over the angle theta0, the resonance strength
    # omegapChiral, and the frequency omega (i.e., output should be 2x100x100),
    # so make the arrays to get this with broadcasting - index order is then
    # itheta0, iomegap, iomega
    omega = np.linspace(1.6, 2.4, 30) # omega in eV
    lambda_vac = 1239.8419843320028 / omega # lambda in nm, "magic" constant is hc in eV*nm
    omegapChiral = np.linspace(0.0, 1.0, 20)[:,None]
    theta0 = np.r_[0, 0.231][:,None,None]

    eps_mol, k_mol = eps_DL(omega, epsinf=2.89, omegap=omegapChiral, omega0=2., gamma=0.05, kappa0=0.0)
    omegaPR = 2.0
    gammaPR = 0.05

    mirror_1 = ts.helicity_preserving_mirror(omega,omegaPR,gammaPR,enantiomer=False)
    mirror_2 = ts.helicity_preserving_mirror(omega,omegaPR,gammaPR,enantiomer=True)
    air_infty = ts.MaterialLayer(d=np.inf, eps=1)
    air_thin  = ts.MaterialLayer(d=0.01,   eps=1)
    molecules = ts.MaterialLayer(d=180.,   eps=eps_mol, kappa=k_mol)

    layers = [air_infty, mirror_1, air_thin, molecules, air_thin, mirror_2, air_infty]
    tScat = ts.TScat(layers, lambda_vac, theta0)

    # np.savez_compressed('tests/test_2.npz', ts=tScat.ts, rs=tScat.rs, td=tScat.td, rd=tScat.rd)
    ref_data = np.load('tests/test_2.npz')
    for k,dat in ref_data.items():
        assert np.allclose(dat, getattr(tScat,k))
