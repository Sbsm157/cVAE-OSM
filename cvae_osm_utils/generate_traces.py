import numpy as np
from .cVAE_OSM_tools import sbox_vectorized


def generate_traces(
    nb_traces,
    nb_samples,
    nb_poi=1,
    mu=0,
    sigmas=np.array([1e-1]),
    targeted_byte=0,
    alpha=np.array([1, 0.5, 1, 0, 2, 0.5, 0.75, 0.25]),
    beta=np.array(
        [1, 2, 0.5, 0.25, 1, 0.15, 1, 0.4, 0.3, 0.6, 0.2, 0.1, 0.25, 0.5, 0.75, 1]
    ),
    scenario="scenario_4",
    isotropic_noise=False,
    key_fixed=False,
    seed=42
):
    """
    Generation of nb_traces traces according to the selected scenario given a fixed key.

    Arguments:
        nb_traces: number of traces to generate
        nb_samples: number of samples per generated trace
        nb_poi: number of points of interest per generated trace
        mu: mean of the multivariate Gaussian distribution which is followed by the white gaussian noise
        sigmas: array of variances of the multivariate Gaussian distribution.
            If the noise is isotropic, len(sigmas)==1.
            Else, noise variances will be randomly chosen among the values in sigmas (hence
            sigmas can have a length smaller than the samples argument).
        targeted_byte: targeted byte
        alpha: coefficients associated with bits leakage (Independent Bit Leakage Model)
        beta: coefficients associated with bits leakage (Multivariate Leakage)
        scenario: name of the scenario used to generate traces
                  - "scenario_1": Hamming Weight leakage model (HW)
                  - "scenario_2": Independent Bit leakage model (IBL)
                  - "scenario_3": Multivariate leakage model, i.e. we consider bits interactions
                  - "scenario_4": Multi leakage model (HW, IBL and multivariate leakage model)
        isotropic_noise: boolean that specifies if the Guassian noise is isotropic
        key_fixed : boolean that specifies if the key is fixed or random
        seed: value of seed for reproductible results. To disable reproductible results option, seed must be set to None

    Returns:
        A tuple composed of:
        - traces: generated traces;
        - plaintexts: used plaintexts;
        - keys: the keys used;
        - targeted_variables: targeted variables (i.e. Sbox output of the xor of plaintexts and the key);
        - sigmas_vector: values of sigmas used to generate these traces.
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
    traces = np.random.multivariate_normal(
        np.array([mu for i in range(nb_samples)]),
        (sigmas_vector**2) * np.identity(nb_samples),
        size=(nb_traces),
    )

    # Initialization of the data (i.e. plaintexts, keys, targeted_variables)
    if key_fixed:
        keys = [0x4a, 0x58, 0x32, 0xae, 0x1f, 0x02, 0x96, 0xe1, 0xcc, 0x3d, 0xb4, 0x13, 0xaa, 0x8c, 0xf6, 0xa7]
    else:
        keys = np.random.randint(0, 256, (nb_traces, 16), np.uint8)

    plaintexts = np.random.randint(0, 256, (nb_traces, 16), np.uint8)
    targeted_variables = np.zeros((nb_traces, 16), dtype="uint8")

    # Computation of the targeted variable (ie. Output Sbox)
    if key_fixed:
        for i in range(16):
            targeted_variables[:, i] = sbox_vectorized(plaintexts[:, i], keys[i])
    else:
        for i in range(16):
            targeted_variables[:, i] = sbox_vectorized(plaintexts[:, i], keys[:, i])

    # Construction of the simulated traces
    monomials_deg_1 = np.unpackbits(
        targeted_variables[:, targeted_byte], bitorder="little"
    ).reshape(nb_traces, 8)

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
