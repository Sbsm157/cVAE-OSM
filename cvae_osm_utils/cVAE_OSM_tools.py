import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

# =====================================================================
#
# --------------------- Computation of the basis ----------------------
#
# =====================================================================


def orthonormal_basis_projection(
    targeted_values, max_nb_monomials_interactions, len_basis
):
    """
    Implementation of the orthonormal basis proposed by Guilley etal. in [GHMR17, Theorem 5].

    [GHMR17]  Sylvain Guilley, Annelie Heuser, Tang Ming, and Olivier Rioul. Stochastic side-channel leakage analysis via orthonormal decomposition.
              In Innovative Security Solutions for Information Technology and Communications: 10th International Conference, SecITC 2017, Bucharest,
              Romania, June 8–9, 2017, Revised Selected Papers 10, pages 12–27. Springer, 2017.

    Arguments:
        targeted_values: set of targeted variables (i.e. Sbox output of the xor of plaintexts and the key)
        max_nb_monomials_interactions : maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered).
                   But it can also be set to another value:
                    (maximal degree of bit interactions=0 => len_basis=1 /
                    maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                    maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                    maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                    maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)

    Returns:
        Projection of target values in Guilley etal. orthonormal basis.
    """

    len_targeted_values = targeted_values.shape[0]
    max_possible_values = 2**max_nb_monomials_interactions

    u = np.asarray(
        [(i, bin(i)[2:].count("1")) for i in range(max_possible_values)],
        dtype=[("val", int), ("hw", int)],
    )
    u_increasing_hw = np.sort(u, order="hw")
    u_increasing_hw = u_increasing_hw["val"]

    # Computation of the Fourier basis using Hadamard matrix [GHMR17, Section 3.2]
    walsh_hadamard_matrix = (1 / 2 ** (max_nb_monomials_interactions / 2)) * hadamard(
        max_possible_values
    )

    # Projection of target values in Fourier basis [GHMR17, Theorem 5]
    basis_projection = np.zeros((len_targeted_values, max_possible_values))
    for i in range(len_targeted_values):
        basis_projection[i, :] = walsh_hadamard_matrix[targeted_values[i], :]

    # Projection sorted by increasing Hamming weight [GHMR17, Section 3.2]
    basis_projection_sorted = basis_projection[:, u_increasing_hw]

    return basis_projection_sorted[:, :len_basis]


def orthonormal_basis_computation_true_coeffs(
    traces, targeted_values, max_nb_monomials_interactions
):
    """
    Implementation of the exact solution for the estimation of the orthonormal basis coefficients [GHMR17, Section 4.1]

    [GHMR17]  Sylvain Guilley, Annelie Heuser, Tang Ming, and Olivier Rioul. Stochastic side-channel leakage analysis via orthonormal decomposition.
              In Innovative Security Solutions for Information Technology and Communications: 10th International Conference, SecITC 2017, Bucharest,
              Romania, June 8–9, 2017, Revised Selected Papers 10, pages 12–27. Springer, 2017.

    Arguments:
        traces: set of traces used to retrieve true basis coefficients
        targeted_values: set of targeted variables (i.e. Sbox output of the xor of plaintexts and the key)
        max_nb_monomials_interactions : maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8

    Returns:
        Estimation of Guilley etal. orthonormal basis coefficients.
    """

    Q = traces.shape[0]
    max_possible_values = 2**max_nb_monomials_interactions
    G = np.zeros((max_possible_values, Q))

    u = np.asarray(
        [(i, bin(i)[2:].count("1")) for i in range(max_possible_values)],
        dtype=[("val", int), ("hw", int)],
    )
    u_increasing_hw = np.sort(u, order="hw")
    u_increasing_hw = u_increasing_hw["val"]

    # Computation of the Fourier basis using Hadamard matrix [GHMR17, Section 3.2]
    walsh_hadamard_matrix = hadamard(max_possible_values) / (
        2 ** (max_nb_monomials_interactions / 2)
    )

    # Computation of G matrix [GHMR17, Section 4.1]
    for i in range(Q):
        G[:, i] = walsh_hadamard_matrix[targeted_values[i], :]

    # Computation of basis coefficients [GHMR17, Proposition 6]
    G_inv = np.linalg.inv(np.dot(G, G.T))
    basis_coeffs = np.dot(traces.T, G.T)
    basis_coeffs = np.dot(basis_coeffs, G_inv)

    # Basis coefficients sorted by increasing Hamming weight
    basis_coeffs_sorted = basis_coeffs[:, u_increasing_hw]

    return basis_coeffs_sorted


# =====================================================================
#
# ------------------------------ Plots --------------------------------
#
# =====================================================================


def weights_visualization(
    encoder, decoder, samples, discard_bias=True, saving=False, path=None, filename=None
):
    """
    Visualization and saving of deterministic parts estimated by encoder and decoder.

    Arguments:
        encoder: encoder of cVAE-OSM model
        decoder: decoder of cVAE-OSM model
        samples: list of samples to display
        discard_bias: boolean used to discard bias (i.e. traces modulation) value in the visualization
        saving: boolean used for saving. If saving is set to True, path and filename must be specified
        path: path for plots saving
        filename: filename for plots saving
    """

    plt.rcParams["text.usetex"] = True

    encoder_weights = encoder.get_layer("psi_layer_encoder").weights[0].numpy()
    decoder_weights = decoder.get_layer("psi_layer_decoder").weights[0].numpy()

    # Visualization of encoder weights
    fig1, ax1 = plt.subplots()
    for i in samples:
        label = "t" + str(i) + "_enc"
        ax1.plot(encoder_weights[discard_bias:, i], label=label)

    ax1.set_title(r"Encoder - $\hat{\Psi}_\phi$ layer weights visualization")
    ax1.set_xlabel("Orthonormal basis")
    ax1.set_ylabel("Coefficients of the orthonormal basis")

    # Visualization of encoder weights
    fig2, ax2 = plt.subplots()
    for i in samples:
        label = "t" + str(i) + "_dec"
        ax2.plot(decoder_weights[discard_bias:, i], label=label)

    ax2.set_title(r"Decoder - $\hat{\Psi}_\theta$ layer weights visualization")
    ax2.set_xlabel("Orthonormal basis")
    ax2.set_ylabel("Coefficients of the orthonormal basis")

    # Saving
    if saving:
        fig1.savefig(path + filename + "_encoder.pdf")
        fig2.savefig(path + filename + "_decoder.pdf")
    else:
        plt.show()


# =====================================================================
#
# ----------------------------- Other ---------------------------------
#
# =====================================================================

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])


def output_sbox(pt, key):
    return AES_Sbox[pt ^ key]


# Vectorization of Sbox output computation
sbox_vectorized = np.vectorize(output_sbox)


def save_cVAE(models, path, filename):
    """
    Saving of cVAE-OSM models parameters (variance estimated by encoder and deterministic parts retrieved by both encoder and decoder).

    Arguments:
        models: list of cVAE-OSM models
        path: path for models saving
        filename: filename for models saving
    """

    psi_layers_enc = np.array(
        [m.encoder.get_layer("psi_layer_encoder").weights[0].numpy() for m in models]
    )
    var_enc = np.array(
        [m.encoder.get_layer("z_mean").weights[0].numpy() for m in models]
    )
    psi_layers_dec = np.array(
        [m.decoder.get_layer("psi_layer_decoder").weights[0].numpy() for m in models]
    )

    # Saving of deterministic parts and variance for each cVAE-OSM model
    np.savez(path + filename + ".npz", psi_layers_enc, var_enc, psi_layers_dec)
