import numpy as np
import scipy.stats as ss
from .cVAE_OSM_tools import sbox_vectorized

#=====================================================================
#
#----- Auxiliary function used for Mutual Information computation ----
#
#=====================================================================


def compute_prob_targeted_variables(z, scenario="scenario_4", alpha=np.array([1,0.5,1,0,2,0.5,0.75,0.25]),\
                 beta=np.array([1, 2, 0.5, 0.25, 1, 0.15, 1, 0.4, 0.3, 0.6, 0.2, 0.1, 0.25, 0.5, 0.75, 1])):
    """
        This function aims at computing the probability of targeted variables given a leakage model.

        Arguments:
            z: targeted variables
            scenario: name of the scenario used
                      - "scenario_1": Hamming Weight leakage model (HW)
                      - "scenario_2": Independent Bit leakage model (IBL)
                      - "scenario_3": Multivariate leakage model, i.e. we consider bits interactions
                      - "scenario_4": Multi leakage model (HW, IBL and multivariate leakage model)
            alpha: coefficients associated with bits leakage (Independent Bit Leakage Model)
            beta: coefficients associated with bits leakage (Multivariate Leakage)

        Raises:
            Exception: if scenario is undefined i.e. scenario != 'scenario_1' and 'scenario_2' and 'scenario_3' and 'scenario_4'

        Returns:
            - Probability of targeted variables Z given a leakage model.
    """
    nb_traces = z.shape[0]

    monomials_deg_1 = np.unpackbits(z,bitorder='little').reshape(nb_traces,8)

    # Computing leakage models
    ## HW
    HW_Y = np.sum(monomials_deg_1, axis=1)

    ## IBL
    IBL_Y = np.sum(monomials_deg_1 * alpha, axis=1)

    ## Multivariate
    leakage = np.array([monomials_deg_1[:,0], monomials_deg_1[:,5], monomials_deg_1[:,6], \
           
        monomials_deg_1[:,1] ^ monomials_deg_1[:,3], monomials_deg_1[:,2] ^ monomials_deg_1[:,4], \
        monomials_deg_1[:,4] ^ monomials_deg_1[:,7], \
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6], monomials_deg_1[:,1] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
        monomials_deg_1[:,1] ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,6],\
        monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6],\
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6] \
        ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,5] \
        ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7]]).T

    multivariate_leakage_Y = np.sum(leakage*beta,axis=1)

    match scenario:
        case "scenario_1":
            # Leakage model (HW only)
            leakages = [HW_Y]

        case "scenario_2":
            # Leakage model (IBL only)
            leakages = [IBL_Y]

        case "scenario_3":
            # Leakage model (multivariate leakage only)
            leakages = [multivariate_leakage_Y]

        case "scenario_4":
            # Leakage model (HW, IBL and multivariate leakage depending on PoIs)
            leakages = [HW_Y, IBL_Y, multivariate_leakage_Y]

        case _:
            raise Exception("Undefined scenario")

    # Computation of P(Z) when considering a HW, IBL or multivariate leakage model
    if len(leakages) == 1:
        classes = np.unique(leakages[0])
        nb_classes = classes.shape[0]
        p_classes = {z_class : np.sum(leakages[0]==z_class)/nb_traces for z_class in classes}
        p_z = np.array([p_classes[leakages[0][i]] for i in range(nb_traces)])
    
    # Computation of P(Z) when considering a multiple leakage model i.e HW, IBL and multivariate leakage model
    else:
        multi_leakage = np.column_stack((leakages[0], leakages[1], leakages[2]))
        classes = np.unique(multi_leakage,axis=0)
        nb_classes = classes.shape[0]
        p_classes = {(z_class[0], z_class[1], z_class[2]): np.sum(np.all(multi_leakage == z_class, axis=1))/nb_traces \
                        for z_class in classes}
        p_z = np.array([p_classes[(multi_leakage[i,0], multi_leakage[i,1], multi_leakage[i,2])] for i in range(nb_traces)])

    return p_z


#=====================================================================
#
#-------------- Computation of the Mutual Information ----------------
#
#=====================================================================



def mutual_information(traces, mu, var, p_z, threshold=1e-300):
    """
        Computation of Mutual Information (MI). 
        MI(Z;X) = H(Z) - H(Z|X) with H(Z|X) = - Σ Pr(Z|X) log2(Pr(Z|X)).
        To compute H(Z|X), this function implements [Mas20, Algorithm 1].

        [Mas20]  Loïc Masure. Towards a better comprehension of deep learning for side-channel analysis.
                 (Vers une meilleure compréhension de l’apprentissage profond appliqué aux attaques par
                 observations). PhD thesis, Sorbonne University, Paris, France, 2020.
        
        Arguments:
            traces: simulated traces
            mu: mean of traces
            var: variance of traces
            p_z: probability of targeted variables given a leakage model
            threshold: . Defaults to 1e-300. 

        Returns:
            Mutual information between traces and targeted variables 

    """
    # H(Z)
    entropy_z = 8

    # Computation of H(Z|X) following [Mas20, Algorithm 1]
    cov = var * np.identity(traces.shape[1])
    tab_H = np.zeros(traces.shape[0], dtype="float64")
    tab_P = np.zeros((traces.shape[0], 256), dtype="float64")

    # Computation of Pr(X|Z) i.e. the likelihood
    for k in range(256):
        tab_P[:,k] = ss.multivariate_normal.pdf(traces, mu[k,:], cov)
    
    # Computation of Pr(X)
    p_x =  np.sum(tab_P * p_z, axis=1)
    
    # Application of a threshold to avoid null values (avoid division by zero error)
    indexs_x = np.where(p_x == 0)[0]
    p_x[indexs_x] = threshold
    p_xs = np.tile(p_x, (256,1)).T
    
    # Computation of Pr(Z|X) using Bayes' theorem  
    tab_P = (tab_P * p_z) / p_xs
    
    # Application of a threshold to avoid null values in tab_P (for log2)
    indexs_x = np.where(tab_P==0)[0]
    indexs_y = np.where(tab_P==0)[1]
    tab_P[indexs_x, indexs_y] = threshold
    
    # Computation of [Mas20, Equation 17]
    tab_H = - np.sum(tab_P * np.log2(tab_P), axis=1)
    
    # Computation of MI(Z;X) = H(Z) - H(Z|X)
    return entropy_z - np.mean(tab_H)


#=====================================================================
#
#------- Generation of traces for assessing SSS/HDLSS problem --------
#
#=====================================================================


def generate_traces_sss_hdlss_experiment(nb_traces_per_classes, nb_samples, nb_poi=1, mu=0, sigmas=np.array([1e-1]), \
                               alpha=np.array([1,0.5,1,0,2,0.5,0.75,0.25]),\
                               beta=np.array([1, 2, 0.5, 0.25, 1, 0.15, 1, 0.4, 0.3, 0.6, 0.2, 0.1, 0.25, 0.5, 0.75, 1]),\
                                scenario="scenario_4", isotropic_noise=False, seed=42):
    """
        Generation of 256 * nb_traces_per_classes traces following the fourth scenario (Multiple leakage models) given a fixed key.
        
        Arguments:
            nb_traces_per_classes: number of traces per classes to generate
            nb_samples: number of samples per generated trace
            nb_poi: number of points of interest per generated trace
            mu: mean of the multivariate Gaussian distribution which is followed by the white gaussian noise
            sigma: array of variances of the multivariate Gaussian distribution. If the noise is isotropic, len(sigma)==1.
            alpha: coefficients associated with bits leakage (Independent Bit Leakage Model)
            beta: coefficients associated with bits leakage (Multivariate Leakage)
            scenario: name of the scenario used to generate traces
                      - "scenario_1": Hamming Weight leakage model (HW)
                      - "scenario_2": Independent Bit leakage model (IBL)
                      - "scenario_3": Multivariate leakage model, i.e. we consider bits interactions
                      - "scenario_4": Multi leakage model (HW, IBL and multivariate leakage model)
            isotropic_noise: boolean that specifies if the Guassian noise is isotropic
            seed: value of seed for reproductible results
        
        Returns:
            - Traces generated.
            - Key, plaintexts, targeted variables (i.e. Sbox output of the xor of plaintexts and the key) and sigmas
              used to generate these traces.
    """

    # Set seed
    np.random.seed(seed)

    # Initialization of the generated traces
    if isotropic_noise:
        if not isinstance(sigmas, list):
            sigmas = [sigmas]
        assert len(sigmas) == 1
        sigmas_vector = np.array([sigmas[0] for i in range(nb_samples)])
    else:
        sigmas_vector = np.random.choice(sigmas, nb_samples)
    
    # Generation of the multivariate Gaussian noise
    traces = np.random.multivariate_normal(np.array([mu for i in range(nb_samples)]), (sigmas_vector**2)*np.identity(nb_samples), \
                                                    size=(256*nb_traces_per_classes)) 

    # Initialization of the data (i.e. plaintexts, keys, targeted_variables)
    keys = [0x4a]
    
    plaintexts = np.array([[i for j in range(nb_traces_per_classes)] for i in range(256)])
    plaintexts = plaintexts.reshape(-1)

    # Computation of the targeted variable (ie. Output Sbox)
    targeted_variables = sbox_vectorized(plaintexts,keys[0])    
    targeted_variables = targeted_variables.astype('uint8')

    # Construction of the simulated traces
    monomials_deg_1 = np.unpackbits(targeted_variables[:], bitorder='little').reshape(256*nb_traces_per_classes,8)

    # Computing leakage models
    ## HW
    HW_Y = np.sum(monomials_deg_1, axis=1)

    ## IBL
    IBL_Y = np.sum(monomials_deg_1 * alpha, axis=1)

    ## Multivariate
    leakage = np.array([monomials_deg_1[:,0], monomials_deg_1[:,5], monomials_deg_1[:,6], \
           
        monomials_deg_1[:,1] ^ monomials_deg_1[:,3], monomials_deg_1[:,2] ^ monomials_deg_1[:,4], \
        monomials_deg_1[:,4] ^ monomials_deg_1[:,7], \
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6], monomials_deg_1[:,1] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
        monomials_deg_1[:,1] ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,6],\
        monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6],\
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6] \
        ^ monomials_deg_1[:,7],\
                
        monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,5] \
        ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7]]).T

    multivariate_leakage_Y = np.sum(leakage*beta,axis=1)

    match scenario:
        case "scenario_1":
            # Leakage model (HW only)
            leakages = [HW_Y]

        case "scenario_2":
            # Leakage model (IBL only)
            leakages = [IBL_Y]

        case "scenario_3":
            # Leakage model (multivariate leakage only)
            leakages = [multivariate_leakage_Y]

        case "scenario_4":
            # Leakage model (HW, IBL and multivariate leakage depending on PoIs)
            leakages = [HW_Y, IBL_Y, multivariate_leakage_Y]

        case _:
            raise Exception("Undefined scenario")
    
    # Insertion of the leakage model in the simulated traces
    # if more than a single leakage model, then PoI l will use
    # the leakage model at position l%len(leakages)
    for l in range(nb_poi):
        if nb_poi == nb_samples:
            traces[:, l] += leakages[l % len(leakages)]
        else:
            traces[:, (int(nb_samples / (nb_poi + 1)) * (l + 1))] += leakages[
                l % len(leakages)
            ]

    return traces, plaintexts, keys, targeted_variables, sigmas_vector