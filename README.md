# Introduction

TSCAT is a Python library using the transfer matrix approach for calculating scattering properties of multilayer structures including chiral materials, and allowing for the inclusion of arbitrary optical elements (such as metamaterial mirrors) that are defined by their transfer matrix (assumed to be calculated/modeled externally).

# Installation

Install the package with `pip`:
```bash
pip install tscat
```

# Usage
All the following examples assume that the following modules are imported:
```python
import tscat as ts
import numpy as np
import matplotlib.pyplot as plt
```

## Simple example
An extremely simple example (a 100nm dielectric layer surrounded by air) is given by the following:
```python
air_infty = ts.MaterialLayer(d=np.inf, eps=1, kappa=0, mu=1)
dielectric = ts.MaterialLayer(d=100., eps=2.25, kappa=0, mu=1)
layers = [air_infty, dielectric, air_infty]
omega = [0.5, 1.0, 1.5]
theta0 = 0
tScat = ts.TScat(layers, omega, theta0)
```

Here, the elements of `layers` can be either:
- `MaterialLayer` objects representing a uniform material layer, which are constructed with the following arguments:
    - `d`: thickness of the layer (in nm)
    - `eps`: dielectric constant of the layer
    - `kappa`: Pasteur chirality parameter of the layer (default: 0)
    - `mu`: magnetic permeability of the layer (default: 1)
- `TransferMatrixLayer` objects representing an arbitrary optical element defined by its transfer matrix (assumed to be calculated/modeled externally), and constructed with a single argument:
    - `M`: 4x4 transfer matrix of the optical element.

The "main" class is `TScat`, which calculates the scattering properties of the structure upon instantiation and takes the following arguments:
- `layers`: list describing the layers of the structure
- `omega`: frequency (in eV)
- `theta0`: incidence angle (in radians)

The scattering properties can then be accessed as attributes of the returned object `tScat`, e.g.:
```python
> print(tScat.Tsp)
[0.97666235 0.92385092 0.87460948]
```
Here, `Tsp` is the transmission (`T`) probability from left to right (`s`) for positively polarized circular light (`p`) at the three frequencies. The eight available scattering probabilities are: `Tsp`, `Tsm`, `Tdp`, `Tdm`, `Rsp`, `Rsm`, `Rdp`, `Rdm`, where the letters stand for:
- `T`/`R`: Transmission / Reflection probability
- `s`/`d`: Light propagating from left to right (`sinister`) / right to left (`dexter`)
- `p`/`m`: Positively / negatively circular polarized light

The full scattering amplitudes (and not just probabilities) are also available:
```python
> print(tScat.ts)
[[[0.91032471+3.85610033e-01j 0.00275845-2.67169949e-03j]
  [0.00275845-2.67169949e-03j 0.91032471+3.85610033e-01j]]

 [[0.68037644+6.80320869e-01j 0.00677529-2.17493982e-07j]
  [0.00677529-2.17493982e-07j 0.68037644+6.80320869e-01j]]

 [[0.38357693+8.54040341e-01j 0.00635712+5.65022470e-03j]
  [0.00635712+5.65022470e-03j 0.38357693+8.54040341e-01j]]]
```
which gives the transmission/reflection (`t`/`r`) amplitudes from left to right/right to left (`s`/`d`). In the example above, these are `3×2×2` arrays, where the first index is the frequency, the second is the output polarization (in order `p`/`m`), and the third is the input polarization (same order). Since the amplitudes are not diagonal in polarization, this output is always in matrix form (the probabilities with capital letters above are summed over the output polarization).

## Parameter scans
Importantly, the `TScat` interface is written to follow the [numpy broadcasting conventions](https://numpy.org/doc/stable/user/basics.broadcasting.html), so that scanning over any desired combination of parameters is easy:
```python
# we want to scan over layer thickness d AND frequency omega
# so make them a 51-element 1D and a 101x1 2D array, respectively
# to give you 101x51 2D output arrays
d = np.linspace(500, 1000, 51)
omega = np.linspace(0.5, 1.5, 101)[:,None]

air_infty = ts.MaterialLayer(d=np.inf, eps=1)
dielectric = ts.MaterialLayer(d=d, eps=2.25)
layers = [air_infty, dielectric, air_infty]
theta0 = 0.3
tScat = ts.TScat(layers, omega, theta0)

plt.pcolormesh(d, omega, tScat.Tsp, cmap='turbo', shading='gouraud')
plt.colorbar()
plt.xlabel('Layer thickness (nm)')
plt.ylabel('Frequency (eV)')
plt.tight_layout(pad=0.5)
```
![thickness scan](figs/thickness_scan.png)

## Chiral materials
We now make the dielectric material chiral, with Pasteur chirality parameter `kappa=1e-3`. Since this is small, the transmission for left- and right-circular polarized light are visually indistinguishable, and we instead plot the differential chiral transmission DCT = 2(Tp - Tm)/(Tp + Tm):
```python
d = np.linspace(500, 1000, 51)
omega = np.linspace(0.5, 1.5, 101)[:,None]

air_infty = ts.MaterialLayer(d=np.inf, eps=1)
dielectric = ts.MaterialLayer(d=d, eps=2.25, kappa=1e-3)
layers = [air_infty, dielectric, air_infty]
theta0 = 0.3
tScat = ts.TScat(layers, omega, theta0)

vmax = abs(tScat.DCTs).max()
plt.pcolormesh(d, omega, tScat.DCTs, cmap='coolwarm',
               vmin=-vmax, vmax=vmax, shading='gouraud')
cb = plt.colorbar()
cb.set_label('Differential Chiral Transmission left to right')
plt.xlabel('Layer thickness (nm)')
plt.ylabel('Frequency (eV)')
plt.tight_layout(pad=0.5)
```
![thickness scan DCT](figs/thickness_scan_DCT.png)

## Arbitrary layers defined by their transfer matrix

TSCAT also supports the use of layers that are not just uniform material layers, but, e.g., metamaterials described by a transfer matrix that is externally provided. This is done by passing a `TransferMatrixLayer` object, which is a simple container for the transfer matrix. The transfer matrix must be as a `...×4×4` array, where the `...` indicate an arbitrary number of dimensions that are treated according to broadcasting rules (e.g., these can describe frequency dependence), and the last two dimension describe the transfer matrix, where the 4 entries correspond to (`sp`,`sm`,`dp`,`dm`) waves (where again, `s`/`d` stand for left- and right-going waves, and `p`/`m` correspond to helicity of plus/minus 1). For example, this can be used to describe a helicity-preserving mirror, and the model described in [Phys. Rev. A 107, L021501 (2021)](https://doi.org/10.1103/PhysRevA.107.L021501) is already provided in the code with a separate helper function `helicity_preserving_mirror(omegaPR,gammaPR,omega,reversed=False)` that returns the transfer matrix for a mirror with a helicity-preserving resonance at frequency `omegaPR` and with linewidth `gammaPR`, for frequencies `omega`. The `reversed` argument can be used to reverse the direction of the mirror, as necessary for creating a helicity-preserving cavity.

# To do
- add example with helicity-preserving cavity
- add example with field amplitudes
- add examples of PRA Letter and PRA long paper
