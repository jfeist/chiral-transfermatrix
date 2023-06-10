"""TSCAT, a transfer/scattering matrix approach for achiral and chiral multilayers by Lorenzo Mauro, Jacopo Fregoni, Johannes Feist, and Remi Avriller"""

__version__ = '0.1.0'

__all__ = ['MaterialLayer', 'TransferMatrixLayer', 'TScat', 'chirality_preserving_mirror']

import numpy as np
from numba import jit

#########################
# Helper functions      #
#########################

@jit(nopython=True)
def inv_multi_2x2(A):
    """same calculation as np.linalg.inv(A[...,:2,:2])"""
    Bshape = A.shape[:-2] + (2,2)
    A = A.reshape(-1,A.shape[-2],A.shape[-1])
    B = np.empty((A.shape[0],2,2),dtype=A.dtype)
    for a,b in zip(A,B):
        # adapted from https://github.com/JuliaArrays/StaticArrays.jl/blob/master/src/inv.jl
        idet = 1/(a[0,0]*a[1,1] - a[0,1]*a[1,0])
        b[0] = ( a[1,1]*idet, -a[0,1]*idet)
        b[1] = (-a[1,0]*idet,  a[0,0]*idet)
    return B.reshape(Bshape)

def transfer_matrix(n1, mu1, costhetas_1, n2, mu2, costhetas_2):
    """transfer matrix for an interface from material 1 to 2 (left to right)"""
    et = (n2 / n1) * np.sqrt(mu1 / mu2)  # ratio of impendances
    ratiocos = costhetas_2[...,None,:] / costhetas_1[...,:,None] # ratio of cosines of the structure of matrix
    par_tr = np.array([[1,-1],[-1,1]]) # matrix to change the sign of the matrix elements to fill correctly
    Mt = (et[...,None,None] + par_tr) * (1 + par_tr * ratiocos) / 4 # array of the transmission matrix
    Mr = (et[...,None,None] + par_tr) * (1 - par_tr * ratiocos) / 4 # array of the reflection matrix
    return np.block([[Mt,Mr],[Mr,Mt]])

def calc_dct(Tp, Tm):
    """differential chiral transmission or reflection"""
    return 2 * (Tp - Tm) / (Tp + Tm)  # Tp is the transmission + and Tm the transmission -

#########################
# Main code             #
#########################

class Layer:
    """A base class for layers"""
    pass

class MaterialLayer(Layer):
    """A layer made of a material described by its optical constants and thickness."""
    def __init__(self,n,k,mu,d,name=""):
        self.n = np.atleast_1d(n)
        self.k = np.atleast_1d(k)
        self.mu = np.atleast_1d(mu)
        self.d = np.atleast_1d(d)
        self.name = name

        # REFRACTIVE INDICES OF CHIRAL MEDIUM
        npl = self.n * np.sqrt(self.mu) * (1 + self.k)  # refractive index n+
        npm = self.n * np.sqrt(self.mu) * (1 - self.k)  # refractive index n-
        # nps has indices [input_indices..., polarization]
        self.nps = np.stack((npl,npm),axis=-1)

    def set_costheta(self, nsinthetas):
        # use that cos(asin(x)) = sqrt(1-x**2)
        # costhetas has indices [input_indices..., polarization]
        self.costhetas = np.sqrt(1 - (nsinthetas / self.nps)**2)

    def phase_matrix_diagonal(self, omega):
        """propagation phases across layer (diagonal of a diagonal matrix)"""
        # constant is 1/ħc in units of 1/(eV nm), which converts
        # from omega in eV (i.e., it is really ħω) to k in nm^-1
        kd = 0.005067730716156395 * omega * self.d
        phis = kd[...,None] * self.nps * self.costhetas
        phil = np.concatenate((-phis, phis), axis=-1)  # array of phases
        return np.exp(1j*phil)

    def transfer_matrix(self, prev):
        """transfer matrix from previous layer (on the left) to this one"""
        return transfer_matrix(prev.n, prev.mu, prev.costhetas,
                               self.n, self.mu, self.costhetas)

class TransferMatrixLayer(Layer):
    """A layer with a fixed transfer matrix (assumed to be from and to air)."""
    def __init__(self,M,name=""):
        assert M.shape[-2:] == (4,4)
        self.M = M
        self.n = 1.
        self.mu = 1.
        self.name = name

    def set_costheta(self, nsinthetas):
        self.costhetas = np.sqrt(1 - nsinthetas**2)

    def phase_matrix_diagonal(self, omega):
        return np.ones(self.M.shape[:-1])

    def transfer_matrix(self, prev):
        return self.M

class TScat:
    """A multilayer made of a sequence of layers. Calculates the scattering properties upon instantiation."""
    def __init__(self, theta0, layers, omega):
        self.layers = layers

        # Snell's law means that n*sin(theta) is conserved, these are the incoming values
        nsinthetas = self.layers[0].nps * np.sin(theta0) + 0j
        for l in layers:
            l.set_costheta(nsinthetas)

        # phase propagation factors in each (interior) layer
        self.phas = [l.phase_matrix_diagonal(omega) for l in self.layers[1:-1]]

        # transfer matrices at the interfaces between layers
        self.M12 = [l2.transfer_matrix(l1) for l1,l2 in zip(self.layers, self.layers[1:])]

        # total transfer matrix
        self.M = self.M12[0] # M of the first interface
        # cycle to add a phase and a successive interface
        for a,b in zip(self.phas, self.M12[1:]):
            c = a[...,None] * b # A @ b where A_[...]ij = delta_ij a_[...]j
            self.M = self.M @ c

        # convert from the transfer matrix (connecting amplitudes on the left and
        # right of an interface) to the scattering matrix (connecting incoming and
        # outgoing amplitudes). the matrices are 2x2 blocks, where the index
        # within each block is for left and right circular polarizations
        trp = self.M[..., 0:2, 2:4]  # reflection block upper right
        tr  = self.M[..., 2:4, 0:2]  # reflection block lower left
        ttp = self.M[..., 2:4, 2:4]  # transmission block lower right
        # this calculates the inverse of self.M[..., 0:2, 0:2] (iterating over all but the last two indices)
        tti = inv_multi_2x2(self.M)  # inversion of transmission block upper left
        self.Rs = tr @ tti    # reflection matrix for incidence from the left
        self.Ts = tti         # transmission matrix for incidence from the left
        self.Rd = -tti @ trp  # reflection matrix for incidence from the right
        self.Td = ttp + tr @ self.Rd # transmission matrix for incidence from the right

        # Calculate transmittance, reflectance, and DCT/DCR

        # sum probabilities over polarization of the outgoing field and return
        # with input polarization as first index
        polarization_sums = lambda x: np.moveaxis(np.sum(abs(x)**2, axis=-2), -1, 0)
        self.Tsp, self.Tsm = polarization_sums(self.Ts) # transmittance +/- for incidence from the left
        self.Rsp, self.Rsm = polarization_sums(self.Rs) # reflectance   +/- for incidence from the left
        self.Tdp, self.Tdm = polarization_sums(self.Td) # transmittance +/- for incidence from the right
        self.Rdp, self.Rdm = polarization_sums(self.Rd) # reflectance   +/- for incidence from the right
        self.dct_s = calc_dct(self.Tsp, self.Tsm)  # dct for incidence from the left
        self.dcr_s = calc_dct(self.Rsp, self.Rsm)  # dcr for incidence from the left
        self.dct_r = calc_dct(self.Tdp, self.Tdm)  # dct for incidence from the right
        self.dcr_r = calc_dct(self.Rdp, self.Rdm)  # dcr for incidence from the right

    def field_ampl(self, layer, cinc):
        """Computes the amplitudes of the fields in a given layer (at the
        right end), given the amplitudes of the incoming fields from the
        left."""

        # matrix-vector multiplication for arrays of matrices and vectors
        matvec_mul = lambda A,v: np.einsum('...ij,...j->...i', A, v)

        # get coefficients on the right for input from the left
        # this is just the transmitted field, there is no incoming field from the right
        self.fwd2 = np.zeros(self.M.shape[:-1], dtype=complex)
        self.fwd2[..., 0:2] = matvec_mul(self.Ts,np.atleast_1d(cinc))

        if layer==len(self.layers)-1:
            return self.fwd2

        # now successively apply the transfer matrices from right to left to get
        # the field amplitude on the right of each layer
        self.fwd2 = matvec_mul(self.M12[-1], self.fwd2)
        for a,b in zip(self.phas[layer:][::-1], self.M12[layer:-1][::-1]):
            self.fwd2 *= a
            self.fwd2 = matvec_mul(b, self.fwd2)

        return self.fwd2
#########################################################################################

def chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=False):
    """make a TransferMatrixLayer instance for a chirality-preserving mirror."""
    tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
    rM = abs(tP)
    phase = tP / rM
    tPM = abs(tP)
    t = np.sqrt((1 - np.abs(tPM)**2) / 2)
    pst = np.exp(1j * np.pi / 2)

    tPP_r = tMM_r = tPP_l = tMM_l = t * pst
    rMP_r = rPM_r = - t / phase**2 * pst**3
    rMP_l = rPM_l = t * phase**2 / pst
    if reversed:
        tMP_r = tPM_l = tPM * phase
        tPM_r = tMP_l = 0 * tMP_r

        rMM_r = tPM * pst**4 / phase**3
        rPP_l = - tPM * phase
        rPP_r = rMM_l = 0 * rMM_r
    else:
        tPM_r = tMP_l = tPM * phase
        tMP_r = tPM_l = 0 * tPM_r

        rPP_r = tPM * pst**4 / phase**3
        rMM_l = - tPM * phase
        rMM_r = rPP_l = 0 * rPP_r

    # [tPP_r.shape] x 2 x 2 scattering matrices
    mshape = tPP_r.shape + (2,2)
    t_right = np.column_stack((tPP_r, tMP_r, tPM_r, tMM_r)).reshape(mshape)
    t_left  = np.column_stack((tPP_l, tMP_l, tPM_l, tMM_l)).reshape(mshape)
    r_right = np.column_stack((rPP_r, rMP_r, rPM_r, rMM_r)).reshape(mshape)
    r_left  = np.column_stack((rPP_l, rMP_l, rPM_l, rMM_l)).reshape(mshape)

    # convert from scattering matrix S to transfer matrix M
    Mt = inv_multi_2x2(t_right)  # Inversion of the Jt matrix to construct the submatrix 2x2 for the transmission
    Mr = r_left @ Mt  # Submatrix 2x2 for the reflection
    Mre = -Mt @ r_right  # Submatrix 2x2 for the reflection on the opposite side
    Mte = t_left - Mr @ r_right

    M = np.block([[Mt,Mre],[Mr,Mte]])
    return TransferMatrixLayer(M)
