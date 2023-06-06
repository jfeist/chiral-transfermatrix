
# TSCAT, a transfer/scattering matrix approach for achiral and chiral multilayers
# Authors: Lorenzo Mauro, Jacopo Fregoni, and Johannes Feist


###########
# LIBRARIES
##################
import numpy as np
##################

class MaterialLayer:
    def __init__(self,n,k,mu,d,name=""):
        self.n = np.asarray(n)
        self.k = np.asarray(k)
        self.mu = np.asarray(mu)
        self.d = d

        # REFRACTIVE INDICES OF CHIRAL MEDIUM
        npl = self.n * np.sqrt(self.mu) * (1 + self.k)  # refractive index n+
        npm = self.n * np.sqrt(self.mu) * (1 - self.k)  # refractive index n-
        self.nps = np.column_stack((npl,npm))

    def set_costheta(self, nsinthetas):
        # use that cos(theta) = sqrt(1-sin(theta)**2)
        self.costhetas = np.sqrt(1 - (nsinthetas / self.nps)**2)

    # propagation phase across layer (diagonal of a diagonal matrix)
    def phase_matrix_diagonal(self, omega):
        kd = 0.005067730716156395 * omega * self.d # k in 1/nanometers, omega in eV
        phip = kd[:,None] * self.nps * self.costhetas
        #phim = self.npm * kd * self.costhetas[:,1]  # phase for n-
        phil = np.hstack((-phip, phip))  # array of phases
        return np.exp(1j*phil)

    # transfer matrix from previous layer to this one
    def transfer_matrix(self, prev):
        return transfer_matrix(prev.n, prev.mu, prev.costhetas,
                               self.n, self.mu, self.costhetas)

class TransferMatrixLayer:
    def __init__(self,M,name=""):
        assert M.ndim == 3
        assert M.shape[1:] == (4,4)
        self.M = M
        self.nomega = M.shape[0]
        self.n = 1.
        self.mu = 1.

    def set_costheta(self, nsinthetas):
        self.costhetas = np.sqrt(1 - nsinthetas**2)

    def phase_matrix_diagonal(self, omega):
        return np.ones((self.nomega, 4))
    
    def transfer_matrix(self, prev):
        return self.M

#######################################################################################################################
# CLASS FOR THE CHIRAL/ACHIRAL TRANSFER MATRICES TO COMPUTE TRANSMISSION, REFLECTIONS, DCT, DCR IN A MULTILAYER PROBLEM
#######################################################################################################################
class TScat:  # creation of the class which computes all is necessary to study a multilayer problem.
####################################################
# INITIALIZER OF THE CLASS WITH THE INPUT PARAMETERS
############################################################################################################################################################
    def __init__(self, theta0, layers, omega):  # initializer of the class
        self.layers = layers

        # Snell's law means that n*sin(theta) is conserved, these are the incoming values
        nsinthetas = self.layers[0].nps * np.sin(theta0) + 0j
        for l in layers:
            l.set_costheta(nsinthetas)

        # CORE OF THE CODE WHICH GENERATES THE TRANSMISSIONS, REFLECTIONS AND DCT

        # phase propagation factors in each (interior) layer
        self.phas = [l.phase_matrix_diagonal(omega) for l in self.layers[1:-1]]

        # transfer matrices at the interfaces between layers
        self.M12 = [l2.transfer_matrix(l1) for l1,l2 in zip(self.layers, self.layers[1:])]

        # total transfer matrix
        self.M = self.M12[0] # M of the first interface
        for a,b in zip(self.phas, self.M12[1:]): # cycle to add a phase and a successive interface
            c = a[:,:,None] * b # A @ b where A_wij = delta_ij a_wj
            self.M = self.M @ c

############################################################################################################################
# CONSTRUCTION OF 2X2 TRANSMISSION AND REFLECTION MATRICES (FOR LIGHT PROPAGATION FROM LEFT TO RIGHT AND FROM RIGHT TO LEFT)
############################################################################################################################
        tt  = self.M[:, 0:2, 0:2]  # transmission block upper left
        trp = self.M[:, 0:2, 2:4]  # reflection block upper right
        tr  = self.M[:, 2:4, 0:2]  # reflection block lower left
        ttp = self.M[:, 2:4, 2:4]  # transmission block lower right
        tti = np.linalg.inv(tt)  # inversion of transmission block upper left
        self.Rs = tr @ tti  # reflection matrix for incidence from the left
        self.Ts = tti  # transmission matrix for incidence from the left
        self.Rd = -tti @ trp  # reflection matrix for incidence from the right
        self.Td = ttp - tr @ tti @ trp  # transmission matrix for incidence from the right
############################################################################################################################

######################################################################################################################
# CONSTRUCTION OF TRANSMITTANCE, REFLECTANCE, AND DCT (FOR LIGHT PROPAGATION FROM LEFT TO RIGHT AND FROM RIGHT TO LEFT)
################################################################################################################################
        self.Tdp, self.Tdm = np.sum(abs(self.Td)**2, axis=1).T # transmittance +/- for incidence from the right
        self.Tsp, self.Tsm = np.sum(abs(self.Ts)**2, axis=1).T # transmittance +/- for incidence from the left
        self.Rdp, self.Rdm = np.sum(abs(self.Rd)**2, axis=1).T # reflectance +/- for incidence from the right
        self.Rsp, self.Rsm = np.sum(abs(self.Rs)**2, axis=1).T # reflectance +/- for incidence from the left
        self.dct_s = calc_dct(self.Tsp, self.Tsm)  # dct for incidence from the left
        self.dcr_s = calc_dct(self.Rsp, self.Rsm)  # dcr for incidence from the left
        self.dct_r = calc_dct(self.Tdp, self.Tdm)  # dct for incidence from the right
        self.dcr_r = calc_dct(self.Rdp, self.Rdm)  # dcr for incidence from the right
#################################################################################################################################
###################################################
# COMPUTING FIELDS AMPLITUDES IN AN ARBITRARY LAYER
##############################################################################################################
    def calc_ampl(self, layer, cinc, omega):
        vw_list = np.zeros((len(omega), 4, len(self.layers)), dtype=complex)
        cin = np.ones((len(omega), 4), dtype=complex)

        # input coefficients (the reflections are needed to evaluate the outputs)
        cin[:, 0:2] = cinc
        cin[:, 2:4] = np.einsum("wij,wj->wi", self.Rs, cin[:,0:2])
        vw = np.linalg.solve(self.M, cin) #it is cf!
        vw_list[:,:,-1] = vw
        vw = np.einsum("wij,wj->wi", self.M12[-1], vw)
        vw_list[:,:,-2] = vw
        for i in range(len(self.layers)-2, 0, -1):
            a = self.phas[i-1]
            b = self.M12[i-1]
            c = b * a[:,None,:] # b @ A where A_wij = delta_ij a_wj
            vw = np.einsum("wij,wj->wi", c, vw)
            vw_list[:,:,i-1] = vw
        self.fwd2 = vw_list[:,:,layer]

        return self.fwd2
###############################################################################################################

# transfer matrix for an interface from material 1 to 2 (left to right)
def transfer_matrix(n1, mu1, costhetas_1, n2, mu2, costhetas_2):
    et = (n2 / n1) * np.sqrt(mu1 / mu2)  # ratio of impendances
    ratiocos = costhetas_2[:,None,:] / costhetas_1[:,:,None] # ratio of cosines of the structure of matrix
    par_tr = np.array([[1,-1],[-1,1]]) # matrix to change the sign of the matrix elements to fill correctly
    Mt = (et[:,None,None] + par_tr) * (1 + par_tr * ratiocos) / 4 # array of the transmission matrix
    Mr = (et[:,None,None] + par_tr) * (1 - par_tr * ratiocos) / 4 # array of the reflection matrix
    return np.block([[Mt,Mr],[Mr,Mt]])

#####
# DCT
#################################################################################################
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

##################################################################
# convert scattering matrix S to transfer matrix M
#########################################################################################################################################
def S_to_M(scat): # no mur in this function (no magnetic field)
    Jt, Jte, Jre, Jr = scat

    Mt = np.linalg.inv(Jt)  # Inversion of the Jt matrix to construct the submatrix 2x2 for the transmission
    Mr = Jr @ Mt  # Submatrix 2x2 for the reflection
    Mre = -Mt @ Jre  # Submatrix 2x2 for the reflection on the opposite side
    Mte = Jte - Mr @ Jre

    return np.block([[Mt,Mre],[Mr,Mte]])
####################################################################################################################################

def chirality_preserving_mirror(omegaPR,gammaPR,omega,reversed=False):
    S = chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega,reversed)
    M = S_to_M(S)
    return TransferMatrixLayer(M)
