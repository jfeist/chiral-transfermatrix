
# TSCAT, a transfer/scattering matrix approach for achiral and chiral multilayers
# Authors: Lorenzo Mauro, Jacopo Fregoni, and Johannes Feist

import numpy as np

class Layer:
    """A base class for layers"""
    pass

class MaterialLayer(Layer):
    """A layer made of a material described by its optical constants and thickness."""
    def __init__(self,n,k,mu,d,name=""):
        self.n = np.asarray(n)
        self.k = np.asarray(k)
        self.mu = np.asarray(mu)
        self.d = d
        self.name = name

        # REFRACTIVE INDICES OF CHIRAL MEDIUM
        npl = self.n * np.sqrt(self.mu) * (1 + self.k)  # refractive index n+
        npm = self.n * np.sqrt(self.mu) * (1 - self.k)  # refractive index n-
        # nps has indices [omega, polarization]
        self.nps = np.column_stack((npl,npm))

    def set_costheta(self, nsinthetas):
        # use that cos(asin(x)) = sqrt(1-x**2)
        # costhetas has indices [omega, polarization]
        self.costhetas = np.sqrt(1 - (nsinthetas / self.nps)**2)

    # propagation phases across layer (diagonal of a diagonal matrix)
    def phase_matrix_diagonal(self, omega):
        # constant is 1/ħc in units of 1/(eV nm), which converts
        # from omega in eV (i.e., it is really ħω) to k in nm^-1
        kd = 0.005067730716156395 * omega * self.d
        phis = kd[:,None] * self.nps * self.costhetas
        phil = np.hstack((-phis, phis))  # array of phases
        return np.exp(1j*phil)

    # transfer matrix from previous layer (on the left) to this one
    def transfer_matrix(self, prev):
        return transfer_matrix(prev.n, prev.mu, prev.costhetas,
                               self.n, self.mu, self.costhetas)

class TransferMatrixLayer(Layer):
    """A layer with a fixed transfer matrix (assumed to be from and to air)."""
    def __init__(self,M,name=""):
        assert M.ndim == 3
        assert M.shape[1:] == (4,4)
        self.M = M
        self.nomega = M.shape[0]
        self.n = 1.
        self.mu = 1.
        self.name = name

    def set_costheta(self, nsinthetas):
        self.costhetas = np.sqrt(1 - nsinthetas**2)

    def phase_matrix_diagonal(self, omega):
        return np.ones((self.nomega, 4))

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
        for a,b in zip(self.phas, self.M12[1:]): # cycle to add a phase and a successive interface
            c = a[:,:,None] * b # A @ b where A_wij = delta_ij a_wj
            self.M = self.M @ c

        # convert from the transfer matrix (connecting amplitudes on the left and
        # right of an interface) to the scattering matrix (connecting incoming and
        # outgoing amplitudes). the matrices are 2x2 blocks, where the index
        # within each block is for left and right circular polarizations
        tt  = self.M[:, 0:2, 0:2]  # transmission block upper left
        trp = self.M[:, 0:2, 2:4]  # reflection block upper right
        tr  = self.M[:, 2:4, 0:2]  # reflection block lower left
        ttp = self.M[:, 2:4, 2:4]  # transmission block lower right
        tti = np.linalg.inv(tt)  # inversion of transmission block upper left
        self.Rs = tr @ tti    # reflection matrix for incidence from the left
        self.Ts = tti         # transmission matrix for incidence from the left
        self.Rd = -tti @ trp  # reflection matrix for incidence from the right
        self.Td = ttp - tr @ tti @ trp  # transmission matrix for incidence from the right

        # Calculate transmittance, reflectance, and DCT/DCR

        # the sum is over polarization of the outgoing field
        self.Rsp, self.Rsm = np.sum(abs(self.Rs)**2, axis=1).T # reflectance +/- for incidence from the left
        self.Tsp, self.Tsm = np.sum(abs(self.Ts)**2, axis=1).T # transmittance +/- for incidence from the left
        self.Rdp, self.Rdm = np.sum(abs(self.Rd)**2, axis=1).T # reflectance +/- for incidence from the right
        self.Tdp, self.Tdm = np.sum(abs(self.Td)**2, axis=1).T # transmittance +/- for incidence from the right
        self.dct_s = calc_dct(self.Tsp, self.Tsm)  # dct for incidence from the left
        self.dcr_s = calc_dct(self.Rsp, self.Rsm)  # dcr for incidence from the left
        self.dct_r = calc_dct(self.Tdp, self.Tdm)  # dct for incidence from the right
        self.dcr_r = calc_dct(self.Rdp, self.Rdm)  # dcr for incidence from the right

    def calc_ampl(self, layer, cinc):
        """Computes the amplitudes of the fields in a given layer, given the
        amplitudes of the incoming fields from the left."""

        # get coefficients on the right for input from the left
        # this is just the transmitted field, there is no incoming field from the right
        self.fwd2 = np.zeros(self.M.shape[:2], dtype=complex)
        self.fwd2[:, 0:2] = np.einsum("wij,wj->wi", self.Ts, np.atleast_2d(cinc))

        # now successively apply the transfer matrices from right to left to get field
        self.fwd2 = np.einsum("wij,wj->wi", self.M12[-1], self.fwd2)
        for a,b in zip(self.phas[layer:][::-1], self.M12[layer:-1][::-1]):
            self.fwd2 = np.einsum("wij,wj->wi", b, a*self.fwd2)

        return self.fwd2
#########################################################################################

# transfer matrix for an interface from material 1 to 2 (left to right)
def transfer_matrix(n1, mu1, costhetas_1, n2, mu2, costhetas_2):
    et = (n2 / n1) * np.sqrt(mu1 / mu2)  # ratio of impendances
    ratiocos = costhetas_2[:,None,:] / costhetas_1[:,:,None] # ratio of cosines of the structure of matrix
    par_tr = np.array([[1,-1],[-1,1]]) # matrix to change the sign of the matrix elements to fill correctly
    Mt = (et[:,None,None] + par_tr) * (1 + par_tr * ratiocos) / 4 # array of the transmission matrix
    Mr = (et[:,None,None] + par_tr) * (1 - par_tr * ratiocos) / 4 # array of the reflection matrix
    return np.block([[Mt,Mr],[Mr,Mt]])

# differential chiral transmission (or reflection, same formula)
def calc_dct(Tp, Tm):
    return 2 * (Tp - Tm) / (Tp + Tm)  # Tp is the transmission + and Tm the transmission -

def chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega,reversed=False):
    ngrid = np.ones_like(omega)
    tP =  gammaPR / (1j * (omega - omegaPR) + gammaPR)
    rM = abs(tP)
    phase = tP / rM
    tPM = abs(tP)
    t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
    phit = np.pi / 2
    pst = np.exp(1j * phit)

    tPP_r = t * pst
    tMM_r = t * pst
    tPP_l = t * pst
    tMM_l = t * pst
    if reversed:
        tMP_r = tPM * phase
        tPM_r = 0.0j * ngrid
        tMP_l = 0.0j * ngrid
        tPM_l = tPM * phase
    else:
        tMP_r = 0.0j * ngrid
        tPM_r = tPM * phase
        tMP_l = tPM * phase
        tPM_l = 0.0j * ngrid

    if reversed:
        rPP_r = 0.0j * ngrid
        rMM_r = tPM * pst**4 / phase**3
        rPP_l = - tPM * phase
        rMM_l = 0.0j * ngrid
    else:
        rPP_r = tPM * pst**4 / phase**3
        rMM_r = 0.0j * ngrid
        rPP_l = 0.0j * ngrid
        rMM_l = - tPM * phase
    rMP_r = - t / phase**2 * pst**3
    rPM_r = - t / phase**2 * pst**3
    rMP_l = t * phase**2 / pst
    rPM_l = t * phase**2 / pst

    # nomega x 2 x 2 scattering matrices
    t_right = np.column_stack((tPP_r, tMP_r, tPM_r, tMM_r)).reshape(-1,2,2)
    t_left  = np.column_stack((tPP_l, tMP_l, tPM_l, tMM_l)).reshape(-1,2,2)
    r_right = np.column_stack((rPP_r, rMP_r, rPM_r, rMM_r)).reshape(-1,2,2)
    r_left  = np.column_stack((rPP_l, rMP_l, rPM_l, rMM_l)).reshape(-1,2,2)

    return t_right, t_left, r_right, r_left

# convert scattering matrix S to transfer matrix M
def S_to_M(scat): # no mur in this function (no magnetic field)
    Jt, Jte, Jre, Jr = scat

    Mt = np.linalg.inv(Jt)  # Inversion of the Jt matrix to construct the submatrix 2x2 for the transmission
    Mr = Jr @ Mt  # Submatrix 2x2 for the reflection
    Mre = -Mt @ Jre  # Submatrix 2x2 for the reflection on the opposite side
    Mte = Jte - Mr @ Jre

    return np.block([[Mt,Mre],[Mr,Mte]])

# make a TransferMatrixLayer instance for a chirality-preserving mirror
def chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=False):
    S = chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega,reversed)
    M = S_to_M(S)
    return TransferMatrixLayer(M)
