Tutorial of TSCAT
=================

Here we present a short practical tutorial on the transfer-scattering matrix approach (TSCAT) we developed. The easiest way is to show the code which generates the transmission through a Fabry-Pérot cavity made by Ag mirrors and filled with a chiral layer and the onset of cavity chiral Polaritons. In addition, we add two examples to compute the intensities of the circularly polarized fields in cavity.

Libraries
---------

The necessary Python libraries to run the code and produce the figures are reported below. The source code is called **t****s****c****a****t****.****p****y**.

    ###########
    # LIBRARIES
    ######################################################
    import tscat as ts  # Essential "import" to run TSCAT

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import show
    ######################################################

Dielectric function and Chiral coupling
---------------------------------------

The implementation of the dielectric function *ε*(*ω*) and the Pasteur (chiral) coupling *κ*(*ω*) are presented below. Their expressions can be modified to model other structures depending on the problem the user faces.

    ###################################################################
    # GENERAL DEFINITION OF THE DIELECTRIC FUNCTION AND CHIRAL COUPLING
    ###############################################################################################
    def eps_DL(epsinf, omegap, omega, omega0=0, gamma=0, k0=0):
        eps = epsinf + (omegap**2 / ((omega0**2 - omega**2) - 1j * gamma * omega))  
        # dispersive dielectric function
        n = np.sqrt(eps)
        
        if k0 != 0:
            k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))  
            # chiral coupling
            return eps, n, k
        else:
            return eps, n, k0
    ###############################################################################################

Frequencies and couplings range
-------------------------------

The source code **t****s****c****a****t****.****p****y** contains loops of *ω* in units of eV, thus the user has to specify the range of *ω* and build a grid of *ω*s as indicated below.

    ######################################################################
    # RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
    ######################################################################
    omega = np.linspace(1.6, 2.4, 100)  # Omega in eV
    ngrid = np.ones_like((omega))
    ######################################################################

Preserving mirrors
------------------

TSCAT also works with preserving mirrors or any other metamirrors for which the response is known. For instance, the preserving mirror modelled in https://doi.org/10.1103/PhysRevA.107.L021501 is implemented below. Note that all the coefficients such as **t****P****P****\_****r** or **t****M****P****\_****r** are collected in lists called **t****1****\_****r****i****g****h****t** and so on. All the elements of the scattering matrix are then stored in the list **s****c****a****t****T****O****T** which contains all the scattering matrices for the modelled custom layers. The implementation of the second preserving mirror is analogous.

    scatTOT = list()  # cumulative scattering matrix for all the custom layers

    ###############################################################
    # DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 1 
    ####################################################################
    omegaPR = 2.0
    gammaPR = 0.05

    tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
    rM =  np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
    phase = tP / rM
    tPM = np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
    t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
    phit = np.pi / 2
    pst = np.exp(1j * phit)

    tPP_r = t * pst 
    tMP_r = 0.0j * ngrid
    tPM_r = tPM * phase 
    tMM_r = t * pst

    tPP_l = t * pst 
    tMP_l = tPM * phase 
    tPM_l = 0.0j * ngrid
    tMM_l = t * pst 

    rPP_r = tPM * pst**4 * (1/phase)**3 
    rMP_r = - t * (1 / phase)**2 * (pst**3) 
    rPM_r = - t * (1 / phase)**2 * (pst**3) 
    rMM_r = 0.0j * ngrid

    rPP_l = 0.0j * ngrid
    rMP_l = t * (phase**2) * (1 / pst)
    rPM_l = t * (phase**2) * (1 / pst)
    rMM_l = - tPM * phase

    t1_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
    t1_left = [tPP_l, tMP_l, tPM_l, tMM_l]
    r1_right = [rPP_r, rMP_r, rPM_r, rMM_r]
    r1_left = [rPP_l, rMP_l, rPM_l, rMM_l]

    scatTOT.append([t1_right, t1_left, r1_right, r1_left])
    ###################################################################

Multilayers for Standard FP
---------------------------

The next script shows the multilayer structure for a standard FP cavity, where the oscillator strength of the LHIC medium (denoted as CHIRAL MATERIAL in the code) defined as $\\omega\_{p}\\sqrt{f}$ is varied. Note the structure of the list called **m****a****t****T****O****T**. This list changes in case of a standard or alternative (in this work preserving) FP cavity. At the end of the script, the computed observables are shown: **T****s****p** (*T*<sub>+</sub><sup>→</sup> for light propagation from right to left), **T****s****m** (*T*<sub>−</sub><sup>→</sup>), **R****s****p** (*R*<sub>+</sub><sup>←</sup> for light propagation from right to left), **R****s****m** (*R*<sub>−</sub><sup>←</sup>) and **d****c****t****\_****s** (𝒟𝒞𝒯 for light propagation from right to left). In case the user is interested in computing the same observables but for the opposite light propagation has to type **T****d****p**, **T****d****m**, **R****d****p**, **R****d****m** or **d****c****t****\_****d** respectively. The user can also type **d****c****r****\_****s** (𝒟𝒞ℛ for light propagation from left to right) or **d****c****r****\_****d** (𝒟𝒞ℛ for light propagation from right to left). The letter s stands for sinister (latin word to say “on the left-side”) and d stands for dexter (latin word to say “on the right-side”).

    coupl = np.linspace(0.0, 1.0, 100)

    Tplist = []
    Tmlist = []
    Rplist = []
    Rmlist = []
    DCTlist = []

    for i in range(len(coupl)):
     
        ################
        # INCIDENT ANGLE
        ################
        theta0 = 0.0
        ################
        
        #####
        # AIR
        ##############
        n1 = 1 * ngrid
        mu1 = 1 * ngrid
        k1 = 0 * ngrid
        d1 = np.inf
        ##############
        
        ########
        # MIRROR
        ########################################################################################
        epsinf = 4.77574276
        omegapMirr = 9.48300763
        eps2, n2, k2 = eps_DL(epsinf, omegapMirr, omega, omega0 = 0, gamma = 0.17486845, k0 = 0)
        mu2 = 1 * ngrid
        k2 = 0 * ngrid
        d2 = 30
        ########################################################################################   

        #################
        # CHIRAL MATERIAL
        ##########################################################################################
        epsinf = 2.89
        omegapChiral = coupl[i]
        eps3M, n3, k3 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 1e-3)
        mu3 = 1 * ngrid
        dL = 150
        ########################################################################################## 

        ########
        # MIRROR
        ########################################################################################
        epsinf = 4.77574276
        omegapMirr = 9.48300763
        eps4, n4, k4 = eps_DL(epsinf, omegapMirr, omega, omega0 = 0, gamma = 0.17486845, k0 = 0)
        mu4 = 1 * ngrid
        k4 = 0 * ngrid
        d4 = 30
        ######################################################################################## 

        #####
        # AIR
        ##############
        n5 = 1 * ngrid
        mu5 = 1 * ngrid
        k5 = 0 * ngrid
        d5 = np.inf
        ##############
        
        ########################################
        # ALL THE ARRAYS OF THE INPUT PARAMETERS
        ####################################################
        nTOT = [n1, n2, n3, n4, n5]
        muTOT = [mu1, mu2, mu3, mu4, mu5]
        kTOT = [k1, k2, k3, k4, k5] 
        dTOT = [d1, d2, dL, d4, d5] 
        matTOT = ['air', 'mirr', 'ChiralMat', 'mirr', 'air']
        ####################################################
        
        ######################
        # CALLING OF THE CLASS
        ################################################################
        tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT)  
        ################################################################
        
        Tplist.append(tScat.Tsp)
        Tmlist.append(tScat.Tsm)
        Rplist.append(tScat.Rsp)
        Rmlist.append(tScat.Rsm)
        DCTlist.append(tScat.dct_s)
        
    #############
    # OBSERVABLES
    #######################    
    arr1 = np.array(Tplist)
    arr2 = np.array(Tmlist)
    arr3 = np.array(Rplist)
    arr4 = np.array(Rmlist)
    arr5 = np.array(DCTlist)
    #######################

Multilayers for Preserving FP
-----------------------------

The only difference with respect to the standard layers, presented in the previous section, consists in adding a thin layer of air before and after the modelled metamirrors and to insert the word **C****u****s****t****o****m** in the list called **m****a****t****T****O****T**. All the other words are not case sensitive.

    coupl = np.linspace(0.0, 1.0, 100)

    Tplist = []
    Tmlist = []
    Rplist = []
    Rmlist = []
    DCTlist = []
    DCAlist = []

    for i in range(len(coupl)):
        
        ################
        # INCIDENT ANGLE
        ################
        theta0 = 0
        ################

        ######
        # AIR
        ##############
        n1 = 1 * ngrid
        mu1 = 1 * ngrid
        k1 = 0 * ngrid
        d1 = np.inf
        ##############

        #####################
        # PRESERVING MIRROR 1
        #######################################
        k2 = 0 * ngrid
        mu2 = 1 * ngrid
        n2 = 1 * ngrid
        d2 = 0  # the distance has no influence
        #######################################
        
        #####
        # AIR 
        ###############
        n3 = 1 * ngrid
        mu3 = 1 * ngrid
        k3 = 0 * ngrid
        d3 = 0.01
        ###############

        #################
        # CHIRAL MATERIAL
        #########################################################################################
        epsinf = 2.89
        omegapChiral = coupl[i]
        eps4M, n4, k4 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 0.0)
        mu4 = 1 * ngrid
        k4 = 0 * ngrid
        dL = 180
        ######################################################################################### 
           
        #####
        # AIR 
        ###############
        n5 = 1 * ngrid
        mu5 = 1 * ngrid
        k5 = 0 * ngrid
        d5 = 0.01
        ###############

        #####################
        # PRESERVING MIRROR 2
        ######################################
        k6 = 0 * ngrid
        mu6 = 1 * ngrid
        n6 = 1 * ngrid
        d6 = 0 # the distance has no influence
        ######################################  

        #####
        # AIR
        ###############
        n7 = 1 * ngrid
        mu7 = 1 * ngrid
        k7 = 0 * ngrid
        d7 = np.inf
        ###############

        ########################################
        # ALL THE ARRAYS OF THE INPUT PARAMETERS
        #############################################################################
        nTOT = [n1, n2, n3, n4, n5, n6, n7]
        muTOT = [mu1, mu2, mu3, mu4, mu5, mu6, mu7]
        kTOT = [k1, k2, k3, k4, k5, k6, k7] 
        dTOT = [d1, d2, d3, dL, d5, d6, d7] 
        matTOT = ['air', 'Custom', 'air', 'ChiralMat', 'air', 'Custom', 'air']
        #############################################################################

        ######################
        # CALLING OF THE CLASS 
        #########################################################################
        tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT, scatTOT)  
        #########################################################################
        
        Tplist.append(tScat.Tsp)
        Tmlist.append(tScat.Tsm)
        Rplist.append(tScat.Rsp)
        Rmlist.append(tScat.Rsm)
        DCTlist.append(tScat.dct_s)
        
    #############
    # OBSERVABLES
    ########################    
    arr1 = np.array(Tplist)
    arr2 = np.array(Tmlist)
    arr3 = np.array(Rplist)
    arr4 = np.array(Rmlist)
    arr5 = np.array(DCTlist)
    ########################

2D Map
------

Here, we show a simple example to make a 2D map.

``` python
######
# PLOT
#############################################################################
plt.pcolormesh(coupl, omega, arr1.T, shading = 'gouraud', cmap = 'rainbow') 
plt.xlabel(r"$\hbar\omega_{p}\sqrt{f}\mathrm{[eV]}$", size = 23)
plt.ylabel(r"$\hbar \omega \mathrm{[eV]}$", size = 23)
cbar = plt.colorbar()
cbar.set_label(r'$T_{+}$', labelpad = -10, y = 1.05, rotation = 0, size = 14)
plt.savefig('Tplus_StandardCavity_VarCoupling.pdf', bbox_inches = 'tight')
#############################################################################

show() 
```

Fields in cavity
================

TSCAT is able to compute the amplitudes of the polarized electric fields in a given layer. Here we show two examples of the intensity of the LCP and RCP field when LCP impinges on the cavity made by preserving mirrors. The multilayers structure is similar to the structure presented in Listing \[multPres\] with the difference that here we consider an empty cavity. Before defining all the layers, we define the range of the distance of the intracavity space called **l**, and the list of the amplitudes called **a****m****p****l**. Eventually, the list of the amplitudes is filled by calling a function of the source code: **c****a****l****c****\_****a****m****p****l****(****)**. This function has three arguments: the first is the number of the layer the user wants to study, the second is in the form \[1, 0\] for incoming LCP, \[0, 1\] for incoming RCP and \[1, 1\] for a mixture of both polarizations. The third argument contains the frequency *ω* in units of eV. The list **a****m****p****l** is then reshaped with arguments **l****e****n****(****l****)**, the 4 polarized field intensities **E****l****p** (*E*<sub>→</sub><sup>+</sup>), **E****r****p** (*E*<sub>←</sub><sup>+</sup>), **E****l****m** (*E*<sub>→</sub><sup>−</sup>), **E****r****m** (*E*<sub>←</sub><sup>−</sup>) and the **l****e****n****(****l****)**. Finally, the total contribution of the polarized intensities is collected in **l****c****p** and **r****c****p**.

    l = np.linspace(150, 450, 600)
    ampl = list()

    #############
    # MULTILAYERS
    .......
    ############

        ampl.append(tScat.calc_ampl(2, [0, 1], omega))  # field in cavity for an incoming LCP wave

    #############
    # OBSERVABLES
    #####################################################################################
    ampl2 = np.array(ampl).reshape(len(l), 4, len(omega)) 
    Elp = ampl2[:, 0, :]
    Elm = ampl2[:, 1, :]
    Erp = ampl2[:, 2, :]
    Erm = ampl2[:, 3, :]
    lcp = Elp * Elp.conj() + Erp * Erp.conj()  # total LCP in layer 2 (inside the cavity)
    rcp = Elm * Elm.conj() + Erm * Erm.conj()  # total RCP in layer 2 (inside the cavity)
    #####################################################################################

Figure \[LCPlcp\] shows the case when the cavity is illuminated by LCP light. In this case the LCP light is largely acumulated inside the cavity near the HP region. In contrast, at resonance the cavity is filled only by RCP light when LCP shines on the cavity, due to the efficient cross-convertion of polarization from LCP to RCP. This aspect is shown in Figure \[RCPlcp\].

<embed src="LCPlcp.pdf" />

<embed src="RCPlcp.pdf" />

Complete scripts
================

Here, for completeness, we attach the entire scripts for the transmission through a Fabry-Pérot cavity made by Ag mirrors and filled with a chiral layer, the onset of cavity chiral Polaritons and one example with the amplitudes of the cavity fields.

Script for transmission in a normal FP cavity
---------------------------------------------

    ###########
    # LIBRARIES
    #####################################################
    import tscat as ts  # Essential "import" to run TSCAT

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import show
    #####################################################


    ###################################################################
    # GENERAL DEFINITION OF THE DIELECTRIC FUNCTION AND CHIRAL COUPLING
    ###############################################################################################
    def eps_DL(epsinf, omegap, omega, omega0=0, gamma=0, k0=0):
        eps = epsinf + (omegap**2 / ((omega0**2 - omega**2) - 1j * gamma * omega))  
        # dispersive dielectric function
        n = np.sqrt(eps)
        
        if k0 != 0:
            k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))  
            # chiral coupling
            return eps, n, k

        else:
            return eps, n, k0
    ###############################################################################################


    ######################################################################
    # RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
    ######################################################################
    omega = np.linspace(1.6, 2.4, 100)  # Omega in eV
    ngrid = np.ones_like((omega))
    ######################################################################

    coupl = np.linspace(0.0, 1.0, 100)

    Tplist = []
    Tmlist = []
    Rplist = []
    Rmlist = []
    DCTlist = []

    for i in range(len(coupl)):
     
        ################
        # INCIDENT ANGLE
        ################
        theta0 = 0.0
        ################
        
        #####
        # AIR
        ###############
        n1 = 1 * ngrid
        mu1 = 1 * ngrid
        k1 = 0 * ngrid
        d1 = np.inf
        ###############
        
        ########
        # MIRROR
        ########################################################################################
        epsinf = 4.77574276
        omegapMirr = 9.48300763
        eps2, n2, k2 = eps_DL(epsinf, omegapMirr, omega, omega0 = 0, gamma = 0.17486845, k0 = 0)
        mu2 = 1 * ngrid
        k2 = 0 * ngrid
        d2 = 30
        ########################################################################################   

        #################
        # CHIRAL MATERIAL
        ##########################################################################################
        epsinf = 2.89
        omegapChiral = coupl[i]
        eps3M, n3, k3 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 1e-3)
        mu3 = 1 * ngrid
        dL = 150
        ########################################################################################## 

        ########
        # MIRROR
        ########################################################################################
        epsinf = 4.77574276
        omegapMirr = 9.48300763
        eps4, n4, k4 = eps_DL(epsinf, omegapMirr, omega, omega0 = 0, gamma = 0.17486845, k0 = 0)
        mu4 = 1 * ngrid
        k4 = 0 * ngrid
        d4 = 30
        ######################################################################################## 

        #####
        # AIR
        ###############
        n5 = 1 * ngrid
        mu5 = 1 * ngrid
        k5 = 0 * ngrid
        d5 = np.inf
        ###############
        
        ########################################
        # ALL THE ARRAYS OF THE INPUT PARAMETERS
        ####################################################
        nTOT = [n1, n2, n3, n4, n5]
        muTOT = [mu1, mu2, mu3, mu4, mu5]
        kTOT = [k1, k2, k3, k4, k5] 
        dTOT = [d1, d2, dL, d4, d5] 
        matTOT = ['air', 'mirr', 'ChiralMat', 'mirr', 'air']
        ####################################################
        
        ######################
        # CALLING OF THE CLASS
        ################################################################
        tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT)  
        ################################################################
        
        Tplist.append(tScat.Tsp)
        Tmlist.append(tScat.Tsm)
        Rplist.append(tScat.Rsp)
        Rmlist.append(tScat.Rsm)
        DCTlist.append(tScat.dct_s)
        
    #############
    # OBSERVABLES
    #######################    
    arr1 = np.array(Tplist)
    arr2 = np.array(Tmlist)
    arr3 = np.array(Rplist)
    arr4 = np.array(Rmlist)
    arr5 = np.array(DCTlist)
    #######################

    ######
    # PLOT
    ############################################################################
    plt.pcolormesh(coupl, omega, arr1.T, shading = 'gouraud', cmap = 'rainbow') 
    plt.xlabel(r"$\hbar\omega_{p}\sqrt{f}\mathrm{[eV]}$", size = 23)
    plt.ylabel(r"$\hbar \omega \mathrm{[eV]}$", size = 23)
    cbar = plt.colorbar()
    cbar.set_label(r'$T_{+}$', labelpad = -10, y = 1.05, rotation = 0, size = 14)
    plt.savefig('Tplus_StandardCavity_VarCoupling.pdf', bbox_inches='tight')
    ############################################################################

    show() 

Script for onset of chiral cavity Polaritons
--------------------------------------------

``` python
###########
# LIBRARIES
######################################################
import tscat as ts  # Essential "import" to run TSCAT

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
######################################################


###################################################################
# GENERAL DEFINITION OF THE DIELECTRIC FUNCTION AND CHIRAL COUPLING
################################################################################################
def eps_DL(epsinf, omegap, omega, omega0=0, gamma=0, k0=0):
    eps = epsinf + (omegap**2 / ((omega0**2 - omega**2) - 1j * gamma * omega))  
    # dispersive dielectric function
    n = np.sqrt(eps)
    
    if k0 != 0:
        k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))  
        # chiral coupling
        return eps, n, k

    else:
        return eps, n, k0
#################################################################################################


######################################################################
# RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
######################################################################
omega = np.linspace(1.6, 2.4, 100)
ngrid = np.ones_like((omega))
######################################################################

scatTOT = list()

###############################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 1
######################################################################
omegaPR = 2.0
gammaPR = 0.05

tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
rM =  np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
phase = tP / rM
tPM=np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
phit = np.pi / 2
pst = np.exp(1j * phit)

tPP_r = t * pst 
tMP_r = 0.0j * ngrid
tPM_r = tPM * phase 
tMM_r = t * pst

tPP_l = t * pst 
tMP_l = tPM * phase 
tPM_l = 0.0j * ngrid
tMM_l = t * pst 

rPP_r = tPM * pst**4 * (1 / phase)**3 
rMP_r = - t * (1 / phase)**2 * (pst**3) 
rPM_r = - t * (1 / phase)**2 * (pst**3) 
rMM_r = 0.0j * ngrid

rPP_l = 0.0j * ngrid
rMP_l = t * (phase**2) * (1 / pst)
rPM_l = t * (phase**2) * (1 / pst)
rMM_l = - tPM * phase

t1_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
t1_left = [tPP_l, tMP_l, tPM_l, tMM_l]
r1_right = [rPP_r, rMP_r, rPM_r, rMM_r]
r1_left = [rPP_l, rMP_l, rPM_l, rMM_l]

scatTOT.append([t1_right, t1_left, r1_right, r1_left])
#####################################################################

###############################################################
# DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 2
######################################################################
omegaPR = 2.0
gammaPR = 0.05

tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
rM  = np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
phase = tP / rM
tPM = np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
phit = np.pi / 2
pst = np.exp(1j * phit)

tPP_r = t * pst
tMP_r = tPM * phase
tPM_r = 0.0j * ngrid
tMM_r = t * pst

tPP_l = t * pst
tMP_l = 0.0j * ngrid
tPM_l = tPM * phase
tMM_l = t * pst

rPP_r = 0.0j * ngrid
rMP_r = - t * (1 / phase)**2 * (pst**3) 
rPM_r = - t * (1 / phase)**2 * (pst**3) 
rMM_r = tPM * pst**4 * (1 / phase)**3

rPP_l = - tPM * phase
rMP_l = t * (phase**2) * (1 / pst)
rPM_l = t * (phase**2) * (1 / pst)
rMM_l = 0.0j * ngrid

t2_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
t2_left = [tPP_l, tMP_l, tPM_l, tMM_l]
r2_right = [rPP_r, rMP_r, rPM_r, rMM_r]
r2_left = [rPP_l, rMP_l, rPM_l, rMM_l]

scatTOT.append([t2_right,t2_left,r2_right,r2_left])
###################################################################

coupl = np.linspace(0.0, 1.0, 100)

Tplist = []
Tmlist = []
Rplist = []
Rmlist = []
DCTlist = []
DCAlist = []

for i in range(len(coupl)):
    
    ################
    # INCIDENT ANGLE
    ################
    theta0 = 0
    ################

    ######
    # AIR
    ###############
    n1 = 1 * ngrid
    mu1 = 1 * ngrid
    k1 = 0 * ngrid
    d1 = np.inf
    ###############

    #####################
    # PRESERVING MIRROR 1
    #######################################
    k2 = 0 * ngrid
    mu2 = 1 * ngrid
    n2 = 1 * ngrid
    d2 = 0  # the distance has no influence
    #######################################
    
    #####
    # AIR 
    ###############
    n3 = 1 * ngrid
    mu3 = 1 * ngrid
    k3 = 0 * ngrid
    d3 = 0.01
    ###############

    #################
    # CHIRAL MATERIAL
    #########################################################################################
    epsinf = 2.89
    omegapChiral = coupl[i]
    eps4M, n4, k4 = eps_DL(epsinf, omegapChiral, omega, omega0 = 2.0, gamma = 0.05, k0 = 0.0)
    mu4 = 1 * ngrid
    k4 = 0 * ngrid
    dL = 180
    ######################################################################################### 
       
    #####
    # AIR 
    ###############
    n5 = 1 * ngrid
    mu5 = 1 * ngrid
    k5 = 0 * ngrid
    d5 = 0.01
    ###############

    #####################
    # PRESERVING MIRROR 2
    ######################################
    k6 = 0 * ngrid
    mu6 = 1 * ngrid
    n6 = 1 * ngrid
    d6 = 0 # the distance has no influence
    ######################################  

    #####
    # AIR
    ###############
    n7 = 1 * ngrid
    mu7 = 1 * ngrid
    k7 = 0 * ngrid
    d7 = np.inf
    ###############

    ########################################
    # ALL THE ARRAYS OF THE INPUT PARAMETERS
    #############################################################################
    nTOT = [n1, n2, n3, n4, n5, n6, n7] 
    muTOT = [mu1, mu2, mu3, mu4, mu5, mu6, mu7]
    kTOT = [k1, k2, k3, k4, k5, k6, k7] 
    dTOT = [d1, d2, d3, dL, d5, d6, d7] 
    matTOT = ['air', 'Custom', 'air', 'ChiralMat', 'air', 'Custom', 'air']
    #############################################################################

    ###########################################
    # CALLING OF THE CLASS FOR THE EMPTY CAVITY
    #########################################################################
    tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT, scatTOT)  
    #########################################################################
    
    Tplist.append(tScat.Tsp)
    Tmlist.append(tScat.Tsm)
    Rplist.append(tScat.Rsp)
    Rmlist.append(tScat.Rsm)
    DCTlist.append(tScat.dct_s)
    
#############
# OBSERVABLES
#######################    
arr1 = np.array(Tplist)
arr2 = np.array(Tmlist)
arr3 = np.array(Rplist)
arr4 = np.array(Rmlist)
arr5 = np.array(DCTlist)
#######################

######
# PLOT
#####################################################################################
plt.pcolormesh(coupl, omega, arr5.T, shading = 'gouraud', cmap = 'rainbow') 
plt.xlabel(r"$\hbar\omega_{p}\sqrt{f}\mathrm{[eV]}$", size = 23)
plt.ylabel(r"$\hbar \omega \mathrm{[eV]}$", size = 23)
cbar = plt.colorbar()
cbar.set_label(r'$\mathcal{DCT}$', labelpad = -10, y = 1.05, rotation = 0, size = 14)
plt.savefig('DCT_MetaFilledCavity_VarCoupling.pdf', bbox_inches='tight')
#####################################################################################

show()
```

Script for Figure \[LCPlcp\]
----------------------------

    ###########
    # LIBRARIES
    ######################################################
    import tscat as ts  # Essential "import" to run TSCAT

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.pyplot import show
    ######################################################


    ###################################################################
    # GENERAL DEFINITION OF THE DIELECTRIC FUNCTION AND CHIRAL COUPLING
    ################################################################################################
    def eps_DL(epsinf, omegap, omega, omega0=0, gamma=0, k0=0):
        eps = epsinf + (omegap**2 / ((omega0**2 - omega**2) - 1j * gamma * omega))  
        # dispersive dielectric function
        n = np.sqrt(eps)
        
        if k0 != 0:
            k = k0 * (omegap**2 * omega / (omega0 * ((omega0**2 - omega**2) - 1j * gamma * omega)))  
            # chiral coupling
            return eps, n, k

        else:
            return eps, n, k0
    #################################################################################################


    ######################################################################
    # RANGE OF OMEGA AND CREATION OF THE CORRESPONDING ARRAY FOR THE INPUT
    ######################################################################
    omega = np.linspace(1.8, 2.2, 500)
    ngrid = np.ones_like((omega))
    ######################################################################

    scatTOT = list()

    ###############################################################
    # DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 1
    ######################################################################
    omegaPR = 2.0
    gammaPR = 0.05

    tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
    rM =  np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
    phase = tP / rM
    tPM=np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
    t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
    phit = np.pi / 2
    pst = np.exp(1j * phit)

    tPP_r = t * pst 
    tMP_r = 0.0j * ngrid
    tPM_r = tPM * phase 
    tMM_r = t * pst

    tPP_l = t * pst 
    tMP_l = tPM * phase 
    tPM_l = 0.0j * ngrid
    tMM_l = t * pst 

    rPP_r = tPM * pst**4 * (1 / phase)**3 
    rMP_r = - t * (1 / phase)**2 * (pst**3) 
    rPM_r = - t * (1 / phase)**2 * (pst**3) 
    rMM_r = 0.0j * ngrid

    rPP_l = 0.0j * ngrid
    rMP_l = t * (phase**2) * (1 / pst)
    rPM_l = t * (phase**2) * (1 / pst)
    rMM_l = - tPM * phase

    t1_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
    t1_left = [tPP_l, tMP_l, tPM_l, tMM_l]
    r1_right = [rPP_r, rMP_r, rPM_r, rMM_r]
    r1_left = [rPP_l, rMP_l, rPM_l, rMM_l]

    scatTOT.append([t1_right, t1_left, r1_right, r1_left])
    #####################################################################

    ###############################################################
    # DEFINITION OF THE SCATTERING MATRICES FOR PRESERVING MIRROR 2
    ######################################################################
    omegaPR = 2.0
    gammaPR = 0.05

    tP = gammaPR / (1j * (omega - omegaPR) + gammaPR)
    rM  = np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
    phase = tP / rM
    tPM = np.abs(gammaPR / (1j * (omega - omegaPR) + gammaPR))
    t = np.sqrt((1 - np.abs(tPM)**2) / 2.0)
    phit = np.pi / 2
    pst = np.exp(1j * phit)

    tPP_r = t * pst
    tMP_r = tPM * phase
    tPM_r = 0.0j * ngrid
    tMM_r = t * pst

    tPP_l = t * pst
    tMP_l = 0.0j * ngrid
    tPM_l = tPM * phase
    tMM_l = t * pst

    rPP_r = 0.0j * ngrid
    rMP_r = - t * (1 / phase)**2 * (pst**3) 
    rPM_r = - t * (1 / phase)**2 * (pst**3) 
    rMM_r = tPM * pst**4 * (1 / phase)**3

    rPP_l = - tPM * phase
    rMP_l = t * (phase**2) * (1 / pst)
    rPM_l = t * (phase**2) * (1 / pst)
    rMM_l = 0.0j * ngrid

    t2_right = [tPP_r, tMP_r, tPM_r, tMM_r]  # 2x2 scattering matrices
    t2_left = [tPP_l, tMP_l, tPM_l, tMM_l]
    r2_right = [rPP_r, rMP_r, rPM_r, rMM_r]
    r2_left = [rPP_l, rMP_l, rPM_l, rMM_l]

    scatTOT.append([t2_right,t2_left,r2_right,r2_left])
    ###################################################################

    l = np.linspace(150, 450, 600)
    ampl = list()

    for dist in l:
        
        ################
        # INCIDENT ANGLE
        ################
        theta0 = 0
        ################

        ######
        # AIR
        ###############
        n1 = 1 * ngrid
        mu1 = 1 * ngrid
        k1 = 0 * ngrid
        d1 = np.inf
        ###############

        #####################
        # PRESERVING MIRROR 1
        #######################################
        k2 = 0 * ngrid
        mu2 = 1 * ngrid
        n2 = 1 * ngrid
        d2 = 0  # the distance has no influence
        #######################################
        
        #####
        # AIR 
        ###############
        n3 = 1 * ngrid
        mu3 = 1 * ngrid
        k3 = 0 * ngrid
        d3 = dist
        ###############

        #####################
        # PRESERVING MIRROR 2
        ######################################
        k4 = 0 * ngrid
        mu4 = 1 * ngrid
        n4 = 1 * ngrid
        d4 = 0 # the distance has no influence
        ######################################  

        #####
        # AIR
        ###############
        n5 = 1 * ngrid
        mu5 = 1 * ngrid
        k5 = 0 * ngrid
        d5 = np.inf
        ###############

        ########################################
        # ALL THE ARRAYS OF THE INPUT PARAMETERS
        ##################################################
        nTOT = [n1, n2, n3, n4, n5] 
        muTOT = [mu1, mu2, mu3, mu4, mu5]
        kTOT = [k1, k2, k3, k4, k5] 
        dTOT = [d1, d2, d3, d4, d5] 
        matTOT = ['air', 'Custom', 'air', 'Custom', 'air']
        ##################################################

        ###########################################
        # CALLING OF THE CLASS FOR THE EMPTY CAVITY
        #########################################################################
        tScat = ts.TScat(theta0, nTOT, muTOT, kTOT, dTOT, omega, matTOT, scatTOT)  
        #########################################################################

        ampl.append(tScat.calc_ampl(2, [1,0], omega))  # field in cavity for an incoming LCP wave
        
        
    #############
    # OBSERVABLES
    #####################################################################################    
    ampl2 = np.array(ampl).reshape(len(l), 4, len(omega))       
    Elp = ampl2[:, 0, :]
    Elm = ampl2[:, 1, :]
    Erp = ampl2[:, 2, :]
    Erm = ampl2[:, 3, :]
    lcp = Elp * Elp.conj()+ Erp * Erp.conj()  # total LCP in layer 2 (inside the cavity)
    rcp = Elm * Elm.conj() + Erm * Erm.conj()  # total RCP in layer 2 (inside the cavity)
    #####################################################################################

    ######
    # PLOT
    #####################################################################################
    plt.pcolormesh(l, omega, lcp.T.real, shading = 'gouraud', cmap = 'inferno', norm=mcolors.LogNorm()) 
    plt.xlabel(r"$L \mathrm{[nm]}$", size = 23)
    plt.ylabel(r"$\hbar \omega \mathrm{[eV]}$", size = 23)
    cbar = plt.colorbar()
    cbar.set_label(r'$\left|E^{LCP}_{incav}\right|^2/\left|E^{LCP}_{0}\right|^2$', labelpad = -10, y = 1.1, rotation = 0, size = 14)
    plt.savefig('Emptycav_incavLCP_LCPdriving.pdf', bbox_inches = 'tight')
    ######################################################################################

    show()
