
# TSCAT, a transfer/scattering matrix approach for achiral and chiral multilayers
# Authors: Lorenzo Mauro, Jacopo Fregoni, and Johannes Feist


###########
# LIBRARIES
##################
import numpy as np
##################


#######################################################################################################################
# CLASS FOR THE CHIRAL/ACHIRAL TRANSFER MATRICES TO COMPUTE TRANSMISSION, REFLECTIONS, DCT, DCR IN A MULTILAYER PROBLEM
#######################################################################################################################
class TScat:  # creation of the class which computes all is necessary to study a multilayer problem.
####################################################
# INITIALIZER OF THE CLASS WITH THE INPUT PARAMETERS
############################################################################################################################################################
    def __init__(self, theta0, n, mu, k, d, omega, mat):  # initializer of the class
        for i, strmat in enumerate(mat):
            if not isinstance(strmat, str):
               n[i] = np.ones_like(omega,dtype=complex)
        self.mat = mat # material names or transfer matrices
        self.n = np.asarray(n, dtype=complex)  # array of refractive indices of media
        self.mu = np.asarray(mu, dtype=complex)
        self.k = np.asarray(k, dtype=complex)  # array of chiral parameters of media
        self.d = np.asarray(d)  # array of lengths of media
        self.theta0 = theta0 + 0j  # incident angle of the ray of light on the first interface
#############################################################################################################################################################

#------------------------------------------------------------------------
# CORE OF THE CODE WHICH GENERATES THE TRANSMISSIONS, REFLECTIONS AND DCT
#------------------------------------------------------------------------

#####################################
# REFRACTIVE INDICES OF CHIRAL MEDIUM
###################################################################################
        self.npl = self.n * np.sqrt(self.mu) * (1 + self.k)  # refractive index n+
        self.npm = self.n * np.sqrt(self.mu) * (1 - self.k)  # refractive index n-
###################################################################################

##########################################################
# ARRAY OF THETAS: THE INCIDENT PLUS THE REFRACTIVE THETAS
############################################################################################################################
        thetatp = np.zeros((len(d), len(omega)), dtype=complex)  # array of 0s of dimension d plus the grid of omegas for n+
        thetatm = np.zeros((len(d), len(omega)), dtype=complex)  # array of 0s of dimension d plus the grid of omegas for n-
        thetatp[0, :] = self.theta0  # first entry is the incident angle for theta+
        thetatm[0, :] = self.theta0  # first entry is the incident angle for theta-
        for i in range(len(d) - 1):  # cycle to fill the refractive angles
            thetatp[i+1, :] = thetar(self.npl[i, :], self.npl[i+1, :], thetatp[i, :])
            thetatm[i+1, :] = thetar(self.npm[i, :], self.npm[i+1, :], thetatm[i, :])
        self.thetatp=thetatp
        self.thetatm=thetatm
############################################################################################################################


####################################
# CREATION OF A LIST OF MATRIX PHASE
######################################################################################################################################################################
        self.phas = []  # list of matrix phases
        for j in range(1, len(d) - 1):  # cycle to fill the list with the phase matrix
            if isinstance(mat[j],str): # string containing the material name
                self.phas.append(phase_matrix_diagonal(thetatp[j], thetatm[j], self.npl[j], self.npm[j], omega, d[j]))
            else: # scattering matrix passed directly
                self.phas.append(np.ones((len(omega), 4)))
#######################################################################################################################################################################


##############################################################################
# CREATION OF A LIST OF THE INTERFACE TRANSFER MATRIX WITH THE ARRAY OF THETAS
###########################################################################################################################################################
        self.M12 = []  # list of the matrix of a single interface
        for i in range(len(d) - 1):  # cycle to fill the list with the array of thetas
            if isinstance(mat[i+1],str): # string containing the material name
                theta12 = [thetatp[i+1], thetatm[i+1], thetatp[i], thetatm[i]]
                 # filling the list with the matrix interface
                self.M12.append(transfer_matrix(self.n[i+1], self.n[i], self.mu[i], self.mu[i+1], theta12))
            else: # scattering matrix passed directly
                self.M12.append(mat[i+1])
############################################################################################################################################################


#############################################################
# MULTIPLICATION OF TRANSFER MATRICES FOR MULTIPLE INTERFACES
#######################################################################################
        M = self.M12[0] # S of the single interface
        for i in range(len(d) - 2):  # cycle to add a phase and a successive interface
            a = self.phas[i]
            b = self.M12[i+1]
            c = a[:,:,None] * b # A @ b where A_wij = delta_ij a_wj
            M = M @ c
        self.M = M
#######################################################################################

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
        self.ctd = abs(self.Td)**2
        self.cts = abs(self.Ts)**2
        self.crd = abs(self.Rd)**2
        self.crs = abs(self.Rs)**2
        self.Tdp, self.Tdm = self.ctd.sum(axis=1).T # transmittance +/- for incidence from the right
        self.Tsp, self.Tsm = self.cts.sum(axis=1).T # transmittance +/- for incidence from the left

        self.Rdp, self.Rdm = self.crd.sum(axis=1).T # reflectance +/- for incidence from the right
        self.Rsp, self.Rsm = self.crs.sum(axis=1).T # reflectance +/- for incidence from the left
        self.dct_s = calc_dct(self.Tsp, self.Tsm)  # dct for incidence from the left
        self.dcr_s = calc_dct(self.Rsp, self.Rsm)  # dcr for incidence from the left
        self.dct_r = calc_dct(self.Tdp, self.Tdm)  # dct for incidence from the right
        self.dcr_r = calc_dct(self.Rdp, self.Rdm)  # dcr for incidence from the right
#################################################################################################################################
###################################################
# COMPUTING FIELDS AMPLITUDES IN AN ARBITRARY LAYER
##############################################################################################################
    def calc_ampl(self, layer, cinc, omega):
        vw_list = np.zeros((len(omega), 4, len(self.d)), dtype=complex)
        cin = np.ones((len(omega), 4), dtype=complex)

        # input coefficients (the reflections are needed to evaluate the outputs)
        cin[:, 0:2] = cinc
        cin[:, 2:4] = np.einsum("wij,wj->wi", self.Rs, cin[:,0:2])
        vw = np.linalg.solve(self.M, cin) #it is cf!
        vw_list[:,:,-1] = vw
        vw = np.einsum("wij,wj->wi", self.M12[-1], vw)
        vw_list[:,:,-2] = vw
        for i in range(len(self.d)-2, 0, -1):
            a = self.phas[i-1]
            b = self.M12[i-1]
            c = b * a[:,None,:] # b @ A where A_wij = delta_ij a_wj
            vw = np.einsum("wij,wj->wi", c, vw)
            vw_list[:,:,i-1] = vw
        self.fwd2 = vw_list[:,:,layer]

        return self.fwd2
###############################################################################################################


##################################################
# REFRACTIVE ANGLES OBTAINED USING THE SNELL'S LAW
#########################################################################################################################################
def thetar(n1, n2, theta0):  # n1 is the refractive index of the first medium and n2 is the refractive index of the next medium
    return np.arcsin(n1 * np.sin(theta0) / n2)
########################################################################################################################################

#############################################################
# PHASE MATRIX FOR THE PROPAGATION OF LIGHT INSIDE THE MEDIUM
#################################################################################################################################
def phase_matrix_diagonal(thetap, thetam, npl, npm, omega, d):
    lamb = 1239.841984332002 / omega  # lambda in nanometers, omega in ev
    phip = 2 * np.pi * npl * d * np.cos(thetap) / lamb  # phase for n+
    phim = 2 * np.pi * npm * d * np.cos(thetam) / lamb  # phase for n-
    phil = np.column_stack((-phip, -phim, phip, phim))  # array of phases
    return np.exp(1j*phil)
##################################################################################################################################

########################################
# TRANSFER MATRIX FOR A SINGLE INTERFACE
#############################################################################################################################
def transfer_matrix(n2, n1, mu1, mu2, theta):
    et = (n2 / n1) * np.sqrt(mu1 / mu2)  # ratio of impendances
    theta = np.asarray(theta).T
    ratiocos = np.cos(theta[:,None,0:2]) / np.cos(theta[:,2:4,None]) # ratio of cosines of the structure of matrix
    par_tr = np.array([[1,-1],[-1,1]]) # matrix to change the sign of the matrix elements to fill correctly
    Mt = (et[:,None,None] + par_tr) * (1 + par_tr * ratiocos) / 4 # array of the transmission matrix
    Mr = (et[:,None,None] + par_tr) * (1 - par_tr * ratiocos) / 4 # array of the reflection matrix
    return np.block([[Mt,Mr],[Mr,Mt]])
################################################################################################################################


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

    # 2x2 scattering matrices
    t_right = [tPP_r, tMP_r, tPM_r, tMM_r]
    t_left  = [tPP_l, tMP_l, tPM_l, tMM_l]
    r_right = [rPP_r, rMP_r, rPM_r, rMM_r]
    r_left  = [rPP_l, rMP_l, rPM_l, rMM_l]

    return [t_right, t_left, r_right, r_left]

##################################################################
# convert scattering matrix S to transfer matrix M
#########################################################################################################################################
def S_to_M(scat): # no mur in this function (no magnetic field)
    t_right, t_left, r_right, r_left = scat

    Jr = np.array(r_left).T.reshape(-1,2,2)
    Jt = np.array(t_right).T.reshape(-1,2,2)
    Jre = np.array(r_right).T.reshape(-1,2,2)
    Jte = np.array(t_left).T.reshape(-1,2,2)

    Mt = np.linalg.inv(Jt)  # Inversion of the Jt matrix to construct the submatrix 2x2 for the transmission
    Mr = Jr @ Mt  # Submatrix 2x2 for the reflection
    Mre = -Mt @ Jre  # Submatrix 2x2 for the reflection on the opposite side
    Mte = Jte - Mr @ Jre

    return np.block([[Mt,Mre],[Mr,Mte]])
####################################################################################################################################

def chirality_preserving_mirror_transfermatrix(omegaPR,gammaPR,omega,reversed=False):
    S = chirality_preserving_mirror_scatmat(omegaPR,gammaPR,omega,reversed)
    return S_to_M(S)
