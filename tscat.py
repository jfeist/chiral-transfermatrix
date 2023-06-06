
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
    def __init__(self, theta0, n, mu, k, d, omega, mat):  # initializer of the class ##da estendere per un numero di layer arbitrari
        for i, strmat in enumerate(mat):
            if not isinstance(strmat, str):
               n[i] = np.ones_like(omega,dtype=complex)
        self.mat = mat # material names or scattering matrices
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
            thetatp[i + 1, :] = self.thetar(self.npl[i, :], self.npl[i + 1, :], thetatp[i, :])
            thetatm[i + 1, :] = self.thetar(self.npm[i, :], self.npm[i + 1, :], thetatm[i, :])
        self.thetatp=thetatp
        self.thetatm=thetatm
############################################################################################################################


####################################
# CREATION OF A LIST OF MATRIX PHASE
######################################################################################################################################################################
        self.phas = []  # list of matrix phases
        for j in range(1, len(d) - 1):  # cycle to fill the list with the phase matrix
            if isinstance(mat[j],str): # string containing the material name
                self.phas.append(self.phimat(thetatp[j], thetatm[j], self.npl[j], self.npm[j], omega, d[j]))
            else: # scattering matrix passed directly
                self.phas.append(np.ones((len(omega), 4)))
#######################################################################################################################################################################


##############################################################################
# CREATION OF A LIST OF THE INTERFACE TRANSFER MATRIX WITH THE ARRAY OF THETAS
###########################################################################################################################################################
        self.M12 = []  # list of the matrix of a single interface
        for i in range(len(d) - 1):  # cycle to fill the list with the array of thetas
            if isinstance(mat[i+1],str): # string containing the material name
                theta12 = [thetatp[i + 1], thetatm[i + 1], thetatp[i], thetatm[i]]
                self.M12.append(self.buildmat(self.n[i + 1], self.n[i], self.mu[i], self.mu[i + 1], theta12)) # filling the list with the matrix interface
            else: # scattering matrix passed directly
                self.M12.append(self.buildmatCustom(mat[i+1]))  # filling the list with the Preserving Chiral Mirror interface
############################################################################################################################################################


#############################################################
# MULTIPLICATION OF TRANSFER MATRICES FOR MULTIPLE INTERFACES
#######################################################################################
        S = self.M12[0] # S of the single interface

        for i in range(len(d) - 2):  # cycle to add a phase and a successive interface
            a = self.phas[i]
            b = self.M12[i + 1]
            c = a[:,:,None] * b # A @ b where A_wij = delta_ij a_wj
            S = S @ c
        self.S = S
#######################################################################################


############################################################################################################################
# CONSTRUCTION OF 2X2 TRANSMISSION AND REFLECTION MATRICES (FOR LIGHT PROPAGATION FROM LEFT TO RIGHT AND FROM RIGHT TO LEFT)
############################################################################################################################
        tt  = self.S[:, 0:2, 0:2]  # transmission block upper left
        trp = self.S[:, 0:2, 2:4]  # reflection block upper right
        tr  = self.S[:, 2:4, 0:2]  # reflection block lower left
        ttp = self.S[:, 2:4, 2:4]  # transmission block lower right
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
        self.dct_s = self.calc_dct(self.Tsp, self.Tsm)  # dct for incidence from the left
        self.dcr_s = self.calc_dct(self.Rsp, self.Rsm)  # dcr for incidence from the left
        self.dct_r = self.calc_dct(self.Tdp, self.Tdm)  # dct for incidence from the right
        self.dcr_r = self.calc_dct(self.Rdp, self.Rdm)  # dcr for incidence from the right
#################################################################################################################################


##################################################
# REFRACTIVE ANGLES OBTAINED USING THE SNELL'S LAW
#########################################################################################################################################
    def thetar(self, n1, n2, theta0):  # n1 is the refractive index of the first medium and n2 is the refractive index of the next medium
        return np.arcsin(n1 * np.sin(theta0) / n2)
########################################################################################################################################


#############################################################
# PHASE MATRIX FOR THE PROPAGATION OF LIGHT INSIDE THE MEDIUM
#################################################################################################################################
    def phimat(self, thetap, thetam, npl, npm, omega, d):
        lamb = 1239.841984332002 / omega  # lambda in nanometers, omega in ev
        phip = 2 * np.pi * npl * d * np.cos(thetap) / lamb  # phase for n+
        phim = 2 * np.pi * npm * d * np.cos(thetam) / lamb  # phase for n-
  #      phip =   np.pi * npl * d * np.cos(thetap) / lamb
  #      phim =   np.pi * npm * d * np.cos(thetam) / lamb
        phil = np.column_stack((-phip, -phim, phip, phim))  # array of phases
        return np.exp(1j*phil)
##################################################################################################################################


########################################
# TRANSFER MATRIX FOR A SINGLE INTERFACE
#############################################################################################################################
    def buildmat(self, n2, n1, mu1, mu2, theta):
        et = (n2 / n1) * np.sqrt(mu1 / mu2)  # ratio of impendances
        theta = np.asarray(theta).T
        ratiocos = np.cos(theta[:,None,0:2]) / np.cos(theta[:,2:4,None]) # ratio of cosines of the structure of matrix
        par_tr = np.array([[1,-1],[-1,1]]) # matrix to change the sign of the matrix elements to fill correctly
        Mt = (et[:,None,None] + par_tr) * (1 + par_tr * ratiocos) / 4 # array of the transmission matrix
        Mr = (et[:,None,None] + par_tr) * (1 - par_tr * ratiocos) / 4 # array of the reflection matrix
        return np.block([[Mt,Mr],[Mr,Mt]])
################################################################################################################################


##################################################################
# TRANSFER MATRIX FOR SPIN PRESERVING MIRROR (AT NORMAL INCIDENCE)
#########################################################################################################################################
    def buildmatCustom(self, scat):  # buildmat has no mur in this function (no magnetic field)
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


#####
# DCT
#################################################################################################
    def calc_dct(self, Tp, Tm):
        calcdct = 2 * (Tp - Tm) / (Tp + Tm)  # Tp is the transmission + and Tm the transmission -
        return calcdct
#################################################################################################


###################################################
# COMPUTING FIELDS AMPLITUDES IN AN ARBITRARY LAYER
##############################################################################################################
    def calc_ampl(self, layer, cinc, omega, dlayer=None):
        vw_list = np.zeros((len(omega), 4, len(self.d)), dtype=complex)
        cin = np.ones((len(omega), 4), dtype=complex)

        # input coefficients (the reflections are needed to evaluate the outputs)
        cin[:, 0:2] = cinc
        cin[:, 2:4] = np.einsum("wij,wj->wi", self.Rs, cin[:,0:2])
        vw = np.linalg.solve(self.S, cin) #it is cf!
        assert vw.shape == (len(omega),4)
        vw_list[:,:,-1] = vw
        vw = np.einsum("wij,wj->wi", self.M12[-1], vw)
        vw_list[:,:,-2] = vw
        for i in range(len(self.d)-2,0,-1):
            a = self.phas[i-1]
            b = self.M12[i-1]
            c = b * a[:,None,:] # b @ A where A_wij = delta_ij a_wj
            vw = np.einsum("wij,wj->wi", c, vw)
            vw_list[:,:,i-1] = vw
        self.fwd2 = vw_list[:,:,layer]

        return self.fwd2
###############################################################################################################

