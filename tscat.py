"""tscat is a transfer/scattering matrix package for achiral and chiral multilayers

authors: Lorenzo Mauro, Jacopo Fregoni, Remi Avriller, and Johannes Feist"""

__version__ = '0.1.0'

__all__ = ['MaterialLayer', 'TransferMatrixLayer', 'TScat', 'chirality_preserving_mirror']

import numpy as np
from functools import cached_property

###############################################################################################################
# Helper functions                                                                                            #
###############################################################################################################

# matrix to change the sign of the matrix elements to fill correctly
_par_tr = np.array([[1,-1],[-1,1]])
# matrix to change from circular to linear polarization
_U_circ_to_lin = np.array([[1,1],[1j,-1j]])/np.sqrt(2)

def inv_multi_2x2(A):
    """same calculation as np.linalg.inv(A[...,:2,:2])"""
    assert A.shape[-2] > 1 and A.shape[-1] > 1
    B = np.empty(A.shape[:-2]+(2,2),dtype=A.dtype)
    idet = 1/(A[...,0,0]*A[...,1,1] - A[...,0,1]*A[...,1,0])
    B[...,0,0] =  A[...,1,1]*idet
    B[...,0,1] = -A[...,0,1]*idet
    B[...,1,0] = -A[...,1,0]*idet
    B[...,1,1] =  A[...,0,0]*idet
    return B

def transfer_matrix(eps1, mu1, costhetas_1, eps2, mu2, costhetas_2):
    """transfer matrix for an interface from material 1 to 2"""
    et = np.sqrt((eps2 * mu1) / (eps1 * mu2)) # ratio of impedances
    ratiocos = costhetas_2[...,None,:] / costhetas_1[...,:,None] # ratio of cosines of the structure of matrix
    Mt = (et[...,None,None] + _par_tr) * (1 + _par_tr * ratiocos) / 4 # array of the transmission matrix
    Mr = (et[...,None,None] + _par_tr) * (1 - _par_tr * ratiocos) / 4 # array of the reflection matrix
    return np.block([[Mt,Mr],[Mr,Mt]])

def polarization_sums(x):
    """for amplitudes with indices (...,pol_out,pol_in), sum probabilities over
    output polarization pol_out and return with input polarization pol_in as
    first index"""
    return np.moveaxis(np.sum(abs(x)**2, axis=-2), -1, 0)

def calc_dct(Tp, Tm):
    """differential chiral transmission or reflection"""
    return 2 * (Tp - Tm) / (Tp + Tm)  # Tp is the transmission + and Tm the transmission -

def circ_to_lin(x):
    """convert matrix amplitudes from circular polarization to linear polarization"""
    return _U_circ_to_lin @ x @ _U_circ_to_lin.conj().T


###############################################################################################################
# Main code                                                                                                   #
###############################################################################################################

class Layer:
    """A base class for layers"""
    pass

class MaterialLayer(Layer):
    """A layer made of a material described by its optical constants and thickness."""
    def __init__(self,d,eps,kappa=0,mu=1,name=""):
        self.eps = np.atleast_1d(eps)
        self.kappa = np.atleast_1d(kappa)
        self.mu = np.atleast_1d(mu)
        self.d = np.atleast_1d(d)
        self.name = name

        # REFRACTIVE INDICES OF CHIRAL MEDIUM
        npl = np.sqrt(self.eps*self.mu) * (1 + self.kappa)  # refractive index n+
        npm = np.sqrt(self.eps*self.mu) * (1 - self.kappa)  # refractive index n-
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
        return transfer_matrix(prev.eps, prev.mu, prev.costhetas,
                               self.eps, self.mu, self.costhetas)

class TransferMatrixLayer(Layer):
    """A layer with a fixed transfer matrix (assumed to be from and to air)."""
    def __init__(self,M,name=""):
        assert M.shape[-2:] == (4,4)
        self.M = M
        self.eps = 1.
        self.mu = 1.
        self.name = name

    def set_costheta(self, nsinthetas):
        self.costhetas = np.sqrt(1 - nsinthetas**2)

    def phase_matrix_diagonal(self, omega):
        return np.ones(self.M.shape[:-1])

    def transfer_matrix(self, prev):
        return self.M

###############################################################################################################
# Main class                                                                                                  #
###############################################################################################################
class TScat:
    """A multilayer made of a sequence of layers. Calculates the scattering properties upon instantiation."""
    def __init__(self, layers, omega, theta0):
        omega = np.atleast_1d(omega)
        theta0 = np.atleast_1d(theta0)

        self.layers = layers
        self.omega = omega
        self.theta0 = theta0

        # Snell's law means that n*sin(theta) is conserved, these are the incoming values
        nsinthetas = layers[0].nps * np.sin(theta0) + 0j
        for l in layers:
            l.set_costheta(nsinthetas)

        # phase propagation factors in each (interior) layer
        self.phas = [l.phase_matrix_diagonal(omega) for l in layers[1:-1]]

        # transfer matrices at the interfaces between layers
        self.M12 = [l2.transfer_matrix(l1) for l1,l2 in zip(layers, layers[1:])]

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
        self.rs = tr @ tti    # reflection matrix for incidence from the left
        self.ts = tti         # transmission matrix for incidence from the left
        self.rd = -tti @ trp  # reflection matrix for incidence from the right
        self.td = ttp + tr @ self.rd # transmission matrix for incidence from the right

    ###############################################################################################################
    # Member functions to access transmittance, reflectance, and DCT/DCR                                          #
    ###############################################################################################################
    # These are implemented as cached properties that are computed on first access and then stored in the object.
    # The names below mean the following:
    # T/R: transmittance/reflectance
    # s/d: incidence from the left/right
    # p/m: incoming polarization +/-
    # DCT/DCR: differential chiral transmittance/reflection
    ###############################################################################################################

    # internal helpers
    _Ts = cached_property(lambda self: polarization_sums(self.ts))
    _Rs = cached_property(lambda self: polarization_sums(self.rs))
    _Td = cached_property(lambda self: polarization_sums(self.td))
    _Rd = cached_property(lambda self: polarization_sums(self.rd))

    # external interface
    Tsp, Tsm = property(lambda self: self._Ts[0]), property(lambda self: self._Ts[1])
    Rsp, Rsm = property(lambda self: self._Rs[0]), property(lambda self: self._Rs[1])
    Tdp, Tdm = property(lambda self: self._Td[0]), property(lambda self: self._Td[1])
    Rdp, Rdm = property(lambda self: self._Rd[0]), property(lambda self: self._Rd[1])
    DCTs = cached_property(lambda self: calc_dct(self.Tsp, self.Tsm))
    DCRs = cached_property(lambda self: calc_dct(self.Rsp, self.Rsm))
    DCTd = cached_property(lambda self: calc_dct(self.Tdp, self.Tdm))
    DCRd = cached_property(lambda self: calc_dct(self.Rdp, self.Rdm))

    # linear polarization transmission and reflection amplitudes
    ts_lin = cached_property(lambda self: circ_to_lin(self.ts))
    rs_lin = cached_property(lambda self: circ_to_lin(self.rs))
    td_lin = cached_property(lambda self: circ_to_lin(self.td))
    rd_lin = cached_property(lambda self: circ_to_lin(self.rd))

    _Ts_lin = cached_property(lambda self: polarization_sums(self.ts_lin))
    _Rs_lin = cached_property(lambda self: polarization_sums(self.rs_lin))
    _Td_lin = cached_property(lambda self: polarization_sums(self.td_lin))
    _Rd_lin = cached_property(lambda self: polarization_sums(self.rd_lin))

    Ts_lin_p, Ts_lin_s = property(lambda self: self._Ts_lin[0]), property(lambda self: self._Ts_lin[1])
    Rs_lin_p, Rs_lin_s = property(lambda self: self._Rs_lin[0]), property(lambda self: self._Rs_lin[1])
    Td_lin_p, Td_lin_s = property(lambda self: self._Td_lin[0]), property(lambda self: self._Td_lin[1])
    Rd_lin_p, Td_lin_s = property(lambda self: self._Rd_lin[0]), property(lambda self: self._Rd_lin[1])

    def field_ampl(self, ilayer, cinc):
        """Computes the amplitudes of the fields in a given layer (at the
        right end), given the amplitudes of the incoming fields from the
        left."""

        # matrix-vector multiplication for arrays of matrices and vectors
        matvec_mul = lambda A,v: np.einsum('...ij,...j->...i', A, v)

        # get coefficients on the right for input from the left
        # this is just the transmitted field, there is no incoming field from the right
        self.fwd2 = np.zeros(self.M.shape[:-1], dtype=complex)
        self.fwd2[..., 0:2] = matvec_mul(self.ts,np.atleast_1d(cinc))

        if ilayer==len(self.layers)-1:
            return self.fwd2

        # now successively apply the transfer matrices from right to left to get
        # the field amplitude on the right of each layer
        self.fwd2 = matvec_mul(self.M12[-1], self.fwd2)
        for a,b in zip(self.phas[ilayer:][::-1], self.M12[ilayer:-1][::-1]):
            self.fwd2 *= a
            self.fwd2 = matvec_mul(b, self.fwd2)

        return self.fwd2
###############################################################################################################

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
