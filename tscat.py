
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
        for i in range(len(d)-1):
            strmat = mat[i]
            if strmat == "Custom":
               n[i] = np.ones_like((omega),dtype=complex)
        self.mat = np.array(mat)  # array of magnetic permeabilities of media
        self.n = np.array(n, dtype=complex)  # array of refractive indeces of media
        self.mu = np.array(mu, dtype=complex)
        self.k = np.array(k, dtype=complex)  # array of chiral parameters of media
        self.d = np.array(d)  # array of lengths of media
        self.theta0 = theta0 + 0 * 1j  # incident angle of the ray of light on the first interface
        #global self.d,self.n,omega
#############################################################################################################################################################   

#------------------------------------------------------------------------
# CORE OF THE CODE WHICH GENERATES THE TRANSMISSIONS, REFLECTIONS AND DCT
#------------------------------------------------------------------------

#####################################
# REFRACTIVE INDECES OF CHIRAL MEDIUM
###################################################################################
        self.npl = self.n * np.sqrt(self.mu) * (1 + self.k)  # refractive index n+
        self.npm = self.n * np.sqrt(self.mu)  * (1 - self.k)  # refractive index n-
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
        global phas
        phas = []  # list of matrix phases
        for j in range(1, len(d) - 1):  # cycle to fill the list with the phase matrix
            strmat = mat[j]  # string containing the material name
            if (strmat == "Custom"):   
                IdentMatrix22 = np.zeros((2, 2, len(omega)), dtype=complex)  # creation of an identity 2x2 
                ZerosMatrix22 = np.zeros((2, 2, len(omega)), dtype=complex)  # creation of a matrix of 0s 2x2
                for k in range(len(omega)):
                    IdentMatrix22[0, 0, k] = 1 + 0j
                    IdentMatrix22[0, 1, k] = 0j
                    IdentMatrix22[1, 0, k] = 0j
                    IdentMatrix22[1, 1, k] = 1 + 0j
                IdMatrix = np.zeros((4, 4, len(omega)), dtype=complex)  # identity matrix 4x4 for the single layer
                for l in range(len(omega)):
                    IdMatrix[:, :, l] = np.block(
                            [[IdentMatrix22[:, :, l], ZerosMatrix22[:, :, l]], [ZerosMatrix22[:, :, l], IdentMatrix22[:, :, l]]]  # block matrix for the single layer
                     )
                phas.append(IdMatrix)
            else: 
                phas.append(self.phimat(thetatp[j], thetatm[j], self.npl[j], self.npm[j], omega, d[j], mat[j]))          
#######################################################################################################################################################################  
  

##############################################################################
# CREATION OF A LIST OF THE INTERFACE TRANSFER MATRIX WITH THE ARRAY OF THETAS 
###########################################################################################################################################################        
        global M12
        counter_custom=0
        M12 = []  # list of the matrix of a single interface
        for i in range(len(d) - 1):  # cycle to fill the list with the array of thetas
            strmatsucc = mat[i + 1]  # string containing the following material name
            theta12 = [thetatp[i + 1], thetatm[i + 1], thetatp[i], thetatm[i]]
            if strmatsucc == "Custom":  
                M12.append(self.buildmatCustom(omega,scat[counter_custom]))  # filling the list with the Preserving Chiral Mirror interface
                counter_custom=counter_custom+1
            else:
                M12.append(self.buildmat(self.n[i + 1], self.n[i], self.mu[i], self.mu[i + 1], theta12, omega)) # filling the list with the matrix interface
############################################################################################################################################################
            

#############################################################
# MULTIPLICATION OF TRANSFER MATRICES FOR MULTIPLE INTERFACES
#######################################################################################
        S = M12[0].copy()  # S of the single interface
        a = phas[0]  # first phase matrix

        for i in range(len(d) - 2):  # cycle to add a phase and a successive interface
            a = phas[i]
            b = M12[i + 1]
            #c = M12[i]
            for k in range(len(omega)):  # cycle over the omegas
                S[:, :, k] = S[:, :, k] @ a[:, :, k] @ b[:, :, k]          
#######################################################################################
  
            
############################################################################################################################
# CONSTRUCTION OF 2X2 TRANSMISSION AND REFLECTION MATRICES (FOR LIGHT PROPAGATION FROM LEFT TO RIGHT AND FROM RIGHT TO LEFT)
############################################################################################################################
        Ts = np.zeros((2, 2, len(omega)), dtype=complex)  # 2x2 transmission matrix for propagation from left to right
        Rs = np.zeros((2, 2, len(omega)), dtype=complex)  # 2x2 reflection matrix for propagation from left to right
        Td = np.zeros((2, 2, len(omega)), dtype=complex)  # 2x2 transmission matrix for propagation from right to left
        Rd = np.zeros((2, 2, len(omega)), dtype=complex)  # 2x2 reflection matrix for propagation from right to left
        for k in range(len(omega)):  # cycle over omegas and extraction of the submatrices of S
            tt = S[0:2, 0:2, k]  # trasmission block upper left
            tti = np.linalg.inv(tt)  # inversion of trasmission block upper left
            trp = S[0:2, 2:4, k]  # reflection block upper right
            tr = S[2:4, 0:2, k]  # reflection block lower left
            ttp = S[2:4, 2:4, k]  # transmission block lower right
            Rs[:, :, k] = tr @ tti  # reflection matrix for incidence from the left
            Ts[:, :, k] = tti.copy()  # transmission matrix for incidence from the left
            Rd[:, :, k] = -tti @ trp  # reflection matrix for incidence from the right
            Td[:, :, k] = ttp - tr @ tti @ trp  # transmission matrix for incidence from the right
        self.Rs=Rs
        self.Ts=Ts 
        self.S=S
############################################################################################################################


######################################################################################################################
# CONSTRUCTION OF TRANSMITTANCE, REFLECTANCE, AND DCT (FOR LIGHT PROPAGATION FROM LEFT TO RIGHT AND FROM RIGHT TO LEFT)
################################################################################################################################            
        self.crs, self.cts, self.crd, self.ctd = (  # 2x2 matrices of 0s to construct the conjugates
            np.zeros_like((Rs)),
            np.zeros_like((Rs)),
            np.zeros_like((Rs)),
            np.zeros_like((Rs))
        )
        for k in range(len(omega)):  # cycle over omegas
            for i in range(2): 
                for j in range(2):
                    self.ctd[i, j, k] = Td[i, j, k] * Td[i, j, k].conj()  # conjugate matrix of the transmission from the right
                    self.cts[i, j, k] = Ts[i, j, k] * Ts[i, j, k].conj()  # conjugate matrix of the transmission from the left
                    self.crd[i, j, k] = Rd[i, j, k] * Rd[i, j, k].conj()  # conjugate matrix of the reflection on the right
                    self.crs[i, j, k] = Rs[i, j, k] * Rs[i, j, k].conj()  # conjugate matrix of the reflection on the left
        self.Tdp, self.Tdm, self.Tsp, self.Tsm = (  # arrays of 0s to construct the transmittances
            np.zeros_like((omega)),
            np.zeros_like((omega)),
            np.zeros_like((omega)),
            np.zeros_like((omega))
        )
        self.Tdp = self.ctd[0, 0, :].real + self.ctd[1, 0, :].real  # transmittance + for incidence from the right
        self.Tdm = self.ctd[1, 1, :].real + self.ctd[0, 1, :].real  # transmittance - for incidence from the right
        self.Tsp = self.cts[0, 0, :].real + self.cts[1, 0, :].real  # transmittance + for incidence from the left
        self.Tsm = self.cts[1, 1, :].real + self.cts[0, 1, :].real  # transmittance - for incidence from the left

        self.Rdp, self.Rdm, self.Rsp, self.Rsm = (  # arrays of 0s to construct the reflectances
            np.zeros_like((omega)),
            np.zeros_like((omega)),
            np.zeros_like((omega)),
            np.zeros_like((omega))
        )
        self.Rdp = self.crd[0, 0, :].real + self.crd[1, 0, :].real  # reflectance + for incidence from the right
        self.Rdm = self.crd[1, 1, :].real + self.crd[0, 1, :].real  # reflectance - for incidence from the right
        self.Rsp = self.crs[0, 0, :].real + self.crs[1, 0, :].real  # reflectance + for incidence from the left
        self.Rsm = self.crs[1, 1, :].real + self.crs[0, 1, :].real  # reflectance - for incidence from the left
        self.dct_s = self.calc_dct(self.Tsp, self.Tsm)  # dct for incidence from the left
        self.dcr_s = self.calc_dct(self.Rsp, self.Rsm)  # dcr for incidence from the left
        self.dct_r = self.calc_dct(self.Tdp, self.Tdm)  # dct for incidence from the right
        self.dcr_r = self.calc_dct(self.Rdp, self.Rdm)  # dcr for incidence from the right
        self.dct_s = self.dct_s.real  # dct has to be a real quantity
        self.dct_r = self.dct_r.real  # dct has to be a real quantity
        self.dcr_s = self.dcr_s.real  # dcr has to be a real quantity
        self.dcr_r = self.dcr_r.real  # dcr has to be a real quantity
#################################################################################################################################


##################################################
# REFRACTIVE ANGLES OBTAINED USING THE SNELL'S LAW
#########################################################################################################################################
    def thetar(self, n1, n2, theta0):  # n1 is the refractive index of the first medium and n2 is the refractive index of the next medium
        thetarr = np.arcsin(n1 * np.sin(theta0) / n2)
        return thetarr
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
        vw_list = np.zeros((4, len(self.d), len(omega)), dtype=complex)
        Tmat = np.zeros((4, 4, len(omega)), dtype=complex)
        vw = np.zeros(4, dtype=complex)
        cin = np.ones((4), dtype=complex)
        cf = np.zeros((4), dtype=complex)
        cumul = np.ones((4, 4, len(omega)), dtype=complex)
        dum = M12[-1].copy()
        for k in range(len(omega)):  # input coefficients (the reflections are needed to evaluate the outputs)
            cumul[:, :, k] = dum[: ,:, k]
            cin[0:2] = cinc
            cin[2:4] = self.Rs[:, :, k]@cin[0:2]
            Tmat[:, :, k] = self.S[:, :, k]
            cf[0:2] = np.dot(self.Ts[:, :, k], cin[0:2])
            vw = np.dot(np.linalg.inv(Tmat[:, :, k]), cin) #it is cf!
            vw_list[:,-1,k] = vw 
            vw = np.dot(cumul[:,:,k],vw)       
            vw_list[:,-2,k] = vw
            for i in range(len(self.d)-2,0,-1):
                a = phas[i-1]
                b = M12[i-1]
                cumul[:,:,k] = np.dot(b[:,:,k], a[:,:,k])
                vw = np.dot(cumul[:,:,k], vw)
                vw_list[:,i-1,k] = vw
        self.fwd2 = vw_list[:,layer,:]

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

