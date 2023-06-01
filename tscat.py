
# TSCAT, a transfer/scattering matrix approach for achiral and chiral multilayers
# Authors: Lorenzo Mauro and Jacopo Fregoni


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
    def __init__(self, theta0, n, mu, k, d, omega, mat, scat=0):  # initializer of the class ##da estendere per un numero di layer arbitrari
        for i, strmat in enumerate(mat):
            if strmat == "Custom":
               n[i] = np.ones_like(omega,dtype=complex)
        self.mat = np.asarray(mat)  # array of magnetic permeabilities of media
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
# REFRACTIVE INDECES OF CHIRAL MEDIUM
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
            strmat = mat[j]  # string containing the material name
            if (strmat == "Custom"):
                IdMatrix = np.zeros((len(omega), 4, 4), dtype=complex)  # identity matrix 4x4 for the single layer
                IdMatrix[:,range(4),range(4)] = 1
                self.phas.append(IdMatrix)
            else:
                self.phas.append(self.phimat(thetatp[j], thetatm[j], self.npl[j], self.npm[j], omega, d[j], mat[j]).transpose(2,0,1))
#######################################################################################################################################################################


##############################################################################
# CREATION OF A LIST OF THE INTERFACE TRANSFER MATRIX WITH THE ARRAY OF THETAS
###########################################################################################################################################################
        counter_custom=0
        self.M12 = []  # list of the matrix of a single interface
        for i in range(len(d) - 1):  # cycle to fill the list with the array of thetas
            strmatsucc = mat[i + 1]  # string containing the following material name
            theta12 = [thetatp[i + 1], thetatm[i + 1], thetatp[i], thetatm[i]]
            if strmatsucc == "Custom":
                self.M12.append(self.buildmatCustom(omega,scat[counter_custom]).transpose(2,0,1))  # filling the list with the Preserving Chiral Mirror interface
                counter_custom=counter_custom+1
            else:
                self.M12.append(self.buildmat(self.n[i + 1], self.n[i], self.mu[i], self.mu[i + 1], theta12, omega).transpose(2,0,1)) # filling the list with the matrix interface
############################################################################################################################################################


#############################################################
# MULTIPLICATION OF TRANSFER MATRICES FOR MULTIPLE INTERFACES
#######################################################################################
        S = self.M12[0].copy()  # S of the single interface
        a = self.phas[0]  # first phase matrix

        for i in range(len(d) - 2):  # cycle to add a phase and a successive interface
            a = self.phas[i]
            b = self.M12[i + 1]
            S[...] = S @ a @ b
            #c = M12[i]
            #for k in range(len(omega)):  # cycle over the omegas
            #    S[:, :, k] = S[:, :, k] @ a[:, :, k] @ b[:, :, k]
#######################################################################################


############################################################################################################################
# CONSTRUCTION OF 2X2 TRANSMISSION AND REFLECTION MATRICES (FOR LIGHT PROPAGATION FROM LEFT TO RIGHT AND FROM RIGHT TO LEFT)
############################################################################################################################
        tt = S[:, 0:2, 0:2]  # trasmission block upper left
        tti = np.linalg.inv(tt)  # inversion of trasmission block upper left
        trp = S[:, 0:2, 2:4]  # reflection block upper right
        tr = S[:, 2:4, 0:2]  # reflection block lower left
        ttp = S[:, 2:4, 2:4]  # transmission block lower right
        Rs = tr @ tti  # reflection matrix for incidence from the left
        Ts = tti.copy()  # transmission matrix for incidence from the left
        Rd = -tti @ trp  # reflection matrix for incidence from the right
        Td = ttp - tr @ tti @ trp  # transmission matrix for incidence from the right
        self.Rs=Rs
        self.Ts=Ts
        self.S=S
############################################################################################################################


######################################################################################################################
# CONSTRUCTION OF TRANSMITTANCE, REFLECTANCE, AND DCT (FOR LIGHT PROPAGATION FROM LEFT TO RIGHT AND FROM RIGHT TO LEFT)
################################################################################################################################
        self.ctd = abs(Td)**2
        self.cts = abs(Ts)**2
        self.crd = abs(Rd)**2
        self.crs = abs(Rs)**2
        self.Tdp = self.ctd[:, 0, 0] + self.ctd[:, 1, 0]  # transmittance + for incidence from the right
        self.Tdm = self.ctd[:, 1, 1] + self.ctd[:, 0, 1]  # transmittance - for incidence from the right
        self.Tsp = self.cts[:, 0, 0] + self.cts[:, 1, 0]  # transmittance + for incidence from the left
        self.Tsm = self.cts[:, 1, 1] + self.cts[:, 0, 1]  # transmittance - for incidence from the left

        self.Rdp = self.crd[:, 0, 0] + self.crd[:, 1, 0]  # reflectance + for incidence from the right
        self.Rdm = self.crd[:, 1, 1] + self.crd[:, 0, 1]  # reflectance - for incidence from the right
        self.Rsp = self.crs[:, 0, 0] + self.crs[:, 1, 0]  # reflectance + for incidence from the left
        self.Rsm = self.crs[:, 1, 1] + self.crs[:, 0, 1]  # reflectance - for incidence from the left
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
    def phimat(self, thetap, thetam, npl, npm, omega, d, mat):
        lamb = 1239.841984332002 / omega  # lambda in nanometers, omega in ev
        phip = 2 * np.pi * npl * d * np.cos(thetap) / lamb  # phase for n+
        phim = 2 * np.pi * npm * d * np.cos(thetam) / lamb  # phase for n-
  #      phip =   np.pi * npl * d * np.cos(thetap) / lamb
  #      phim =   np.pi * npm * d * np.cos(thetam) / lamb
        phil = np.array([-phip, -phim, phip, phim])  # array of phases
        Pmat = np.zeros((4, 4, len(omega)), dtype=complex)  # creation of matrix of phases
        for i in range(len(omega)):
                for j in range(4):
                    Pmat[j, j, i] = np.exp(1j * phil[j, i])  # fill the matrix with the elements plus a row of frequencies
        return Pmat
##################################################################################################################################


########################################
# TRANSFER MATRIX FOR A SINGLE INTERFACE
#############################################################################################################################
    def buildmat(self, n2, n1, mu1, mu2, theta, omega):
        et = (n2 / n1) * np.sqrt(mu1 / mu2)  # ratio of impendances
        ratiocos = np.array(
            [
                np.cos(theta[0]) / np.cos(theta[2]),
                np.cos(theta[1]) / np.cos(theta[2]),  # ratio of cosines of the structure of matrix
                np.cos(theta[0]) / np.cos(theta[3]),
                np.cos(theta[1]) / np.cos(theta[3])
            ]
        )
        Mt = np.zeros((4, len(omega)), dtype=complex)  # array of 0s for the transmission
        Mr = np.zeros((4, len(omega)), dtype=complex)  # array of 0s for the reflection
        par_tr = np.ones_like((Mt))  # identity matrix 4x4
        for i in range(len(omega)):  # cycle over the range of omegas
            par_tr[:, i] = par_tr[:, i] * [1, -1, -1, 1]  # matrix to change the sign of the matrix elements to fill correctly
            Mt[:, i] = np.array(
                (et[i] + 1 * par_tr[:, i]) / 4 * (1 + par_tr[:, i] * ratiocos[:, i])  # array of the transmission matrix
            )
            Mr[:, i] = np.array(
                (et[i] + 1 * par_tr[:, i]) / 4 * (1 - par_tr[:, i] * ratiocos[:, i])  # array of the reflection matrix
            )
        Mt = Mt.reshape(2, 2, len(omega))  # matrix 2x2 for the reflection
        Mr = Mr.reshape(2, 2, len(omega))  # matrix 2x2 for the transmission
        M = np.zeros((4, 4, len(omega)), dtype=complex)  # matrix 4x4 of zeros for the single layer
        for i in range(len(omega)):
            M[:, :, i] = np.block(
                    [[Mt[:, :, i], Mr[:, :, i]], [Mr[:, :, i], Mt[:, :, i]]]  # block matrix for the single layer
            )
        return M
################################################################################################################################


##################################################################
# TRANSFER MATRIX FOR SPIN PRESERVING MIRROR (AT NORMAL INCIDENCE)
#########################################################################################################################################
    def buildmatCustom(self, omega, scat):  # buildmat has no mur in this function (no magnetic field)

        Jr = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the reflection
        Jt = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the transmission
        Jre = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the reflection
        Jte = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the transmission
        Mr = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the reflection
        Mt = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the transmission
        Mre = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the reflection
        Mte = np.zeros((2, 2, len(omega)), dtype=complex)  # Jones matrix for the transmission

        t_right = scat[0]
        t_left = scat[1]
        r_right = scat[2]
        r_left = scat[3]

        r1_left1 = r_left[0]
        r1_left2 = r_left[1]
        r1_left3 = r_left[2]
        r1_left4 = r_left[3]

        t1_left1 = t_left[0]
        t1_left2 = t_left[1]
        t1_left3 = t_left[2]
        t1_left4 = t_left[3]

        r1_right1 = r_right[0]
        r1_right2 = r_right[1]
        r1_right3 = r_right[2]
        r1_right4 = r_right[3]

        t1_right1 = t_right[0]
        t1_right2 = t_right[1]
        t1_right3 = t_right[2]
        t1_right4 = t_right[3]


        for k in range(len(omega)):  # cycle over the range of omegas
            Jr[0, 0, k] = r1_left1[k]
            Jr[0, 1, k] = r1_left2[k]
            Jr[1, 0, k] = r1_left3[k]
            Jr[1, 1, k] = r1_left4[k]

            Jt[0, 0, k] = t1_right1[k]
            Jt[0, 1, k] = t1_right2[k]
            Jt[1, 0, k] = t1_right3[k]
            Jt[1, 1, k] = t1_right4[k]

        for k in range(len(omega)):  # cycle over the range of omegas
            Jre[0, 0, k] = r1_right1[k]
            Jre[0, 1, k] = r1_right2[k]
            Jre[1, 0, k] = r1_right3[k]
            Jre[1, 1, k] = r1_right4[k]

            Jte[0, 0, k] = t1_left1[k]
            Jte[0, 1, k] = t1_left2[k]
            Jte[1, 0, k] = t1_left3[k]
            Jte[1, 1, k] = t1_left4[k]


            Mt[:,:,k] = np.linalg.inv(Jt[:,:,k])  # Inversion of the Jt matrix to construct the submatrix 2x2 for the transmission
            Mr[:,:,k] = Jr[:,:,k] @ Mt[:,:,k]  # Submatrix 2x2 for the reflection
            Mre[:,:,k] = -Mt[:,:,k] @ Jre[:,:,k]  # Submatrix 2x2 for the reflection on the opposite side
            Mte[:,:,k] = Jte[:,:,k]-Mr[:,:,k]@Jre[:,:,k]


        M = np.zeros((4, 4, len(omega)), dtype=complex)  # matrix 4x4 of zeros for the single layer
        for i in range(len(omega)):
            M[:, :, i] = np.block(

                    [[Mt[:, :, i], Mre[:, :, i]], [Mr[:, :, i], Mte[:, :, i]]]  # block matrix for the single layer
            )
        return M
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
        Tmat = np.zeros((len(omega), 4, 4), dtype=complex)
        vw = np.zeros(4, dtype=complex)
        cin = np.ones(4, dtype=complex)
        cf = np.zeros(4, dtype=complex)
        cumul = np.ones((len(omega), 4, 4), dtype=complex)
        dum = self.M12[-1].copy()
        assert dum.shape==(len(omega),4,4)
        for k in range(len(omega)):  # input coefficients (the reflections are needed to evaluate the outputs)
            cumul[k, :, :] = dum[k, : ,:]
            cin[0:2] = cinc
            cin[2:4] = self.Rs[k,:, :] @ cin[0:2]
            Tmat[k, :, :] = self.S[k, :, :]
            cf[0:2] = np.dot(self.Ts[k, :, :], cin[0:2])
            vw = np.dot(np.linalg.inv(Tmat[k, :, :]), cin) #it is cf!
            vw_list[k,:,-1] = vw
            vw = np.dot(cumul[k,:,:],vw)
            vw_list[k,:,-2] = vw
            for i in range(len(self.d)-2,0,-1):
                a = self.phas[i-1]
                b = self.M12[i-1]
                cumul[k,:,:] = np.dot(b[k,:,:], a[k,:,:])
                vw = np.dot(cumul[k,:,:], vw)
                vw_list[k,:,i-1] = vw
        self.fwd2 = vw_list[:,:,layer]

        return self.fwd2
#
#
#        if dlayer is not None:
#            last_phas=phas[layer]
#            phas_dist=list()
#            field_dist=list()
#            for k in range(len(omega)):  # input coefficients (the reflections are needed to evaluate the outputs)
#                vw_distance=np.dot(vw_list[:,layer,k],np.linalg.inv(last_phas[:,:,k])) #forse layer-1
#            for dist in range(len(dlayer)):
#                phas_dist.append(self.phimat(self.thetatp[layer],self.thetatm[layer],self.npl[layer],self.npm[layer],omega,dlayer[dist],self.mat[layer]))
#                a=phas_dist[-1]
#            #print(np.shape(phas_dist),np.shape(self.npl[layer]))
#                for k in range(len(omega)):
#                    field_dist.append(np.dot(vw_list[:,layer,k],a[:,:,k]))
#            self.field_dist=field_dist
#            return self.field_dist
#        else:

###############################################################################################################

