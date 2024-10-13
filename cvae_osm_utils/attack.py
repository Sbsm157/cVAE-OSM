import numpy as np
from .cVAE_OSM_tools import sbox_vectorized, orthonormal_basis_projection

# =====================================================================
#
# -------- Computation of the optimal dimensionality reduction --------
#
# =====================================================================


def compute_optimal_dimensionality_reduction(
    traces, projection_targeted_variables, psi_layer, variance_enc
):
    """
    Computation of the optimal dimensionality reduction defined in Theorem 2, Equation 3 (see Section 3.1).

    Arguments:
        traces: set of traces used to compute the optimal dimensionality reduction
        projection_targeted_variables: projection of variable into a basis (see strategy provided in Section 4.2).
                                       In our case we consider projection onto Guilley etal. orthonormal basis [GHMR17]
                                       [GHMR17]  Sylvain Guilley, Annelie Heuser, Tang Ming, and Olivier Rioul. Stochastic
                                                 side-channel leakage analysis via orthonormal decomposition. In Innovative
                                                 Security Solutions for Information Technology and Communications: 10th
                                                 International Conference, SecITC 2017, Bucharest, Romania, June 8–9, 2017,
                                                 Revised Selected Papers 10, pages 12–27. Springer, 2017.
        psi_layer: coefficients of deterministic part estimated by the model
        variance_enc: variance estimated by the model

    Returns:
        The optimal dimensionality reduction of traces
    """

    variance_enc = variance_enc.reshape(-1)

    # Computation of the estimated deterministic part
    estimated_deterministic_part = projection_targeted_variables @ psi_layer

    # Computation of the optimal dimensionality reduction
    opt_dim_red = (traces - estimated_deterministic_part) ** 2 / variance_enc
    opt_dim_red = np.sum(opt_dim_red, axis=1)

    return opt_dim_red


# =====================================================================
#
# ----------------------------- Attack --------------------------------
#
# =====================================================================


def run_attack(
    attack_traces,
    attack_plaintexts,
    psi_layer,
    variance_encoder,
    indexs_pois,
    max_nb_monomials_interactions=8,
    len_basis=256,
    total_nb_attack=10,
    nb_traces=np.arange(1, 101, 1),
    true_key=0x4A,
):
    """
    Runs *total_nb_attack* profiled attacks following the strategy introduced in Section 4.2.

    References mentioned in the code:
    [GHMR17]  Sylvain Guilley, Annelie Heuser, Tang Ming, and Olivier Rioul. Stochastic side-channel leakage analysis via orthonormal decomposition.
              In Innovative Security Solutions for Information Technology and Communications: 10th International Conference, SecITC 2017, Bucharest,
              Romania, June 8–9, 2017, Revised Selected Papers 10, pages 12–27. Springer, 2017.

    Arguments:
        attack_traces: set of attack traces
        attack_plaintexts: set of attack plaintexts
        psi_layer: coefficients of deterministic part estimated by the model
        variance_encoder: variance estimated by the model
        indexs_pois: indexs of points of interest (PoIs)
        max_nb_monomials_interactions: maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered).
                   But it can also be set to another value:
                    (maximal degree of bit interactions=0 => len_basis=1 /
                    maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                    maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                    maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                    maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        total_nb_attack: total number of attacks to carry out
        nb_traces: array that contains numbers of traces on which we conduct attacks
        true_key: real key byte value

    Returns:
        Evolution of mean rank

    """

    indexs_traces = np.arange(attack_traces.shape[0])

    offset_departure_subsets_traces = attack_traces.shape[0] // total_nb_attack

    index_traces_attacks = []

    # Setting attack trace indexes for each attack
    for current_attack in range(total_nb_attack):
        index_traces_attacks.append(
            np.roll(indexs_traces, offset_departure_subsets_traces * current_attack)
        )

    print("Correct key byte:", true_key)

    mean_rank = np.zeros(nb_traces.shape[0])

    # Running of the profiled attack on all numbers of traces
    for current_nb_traces in range(nb_traces.shape[0]):

        keys_evol = []

        # Running of the profiled attack *total_nb_attack* times considering *current_nb_traces* traces (Section 2.4)
        for current_attack in range(total_nb_attack):

            opt_red = np.zeros(256)

            # Extraction of the proper attack traces subset
            index_traces_current_attack = index_traces_attacks[current_attack]
            current_subset_index = index_traces_current_attack[
                : nb_traces[current_nb_traces]
            ]
            traces_attack_subset = attack_traces[current_subset_index, :]

            # Run of the end-to-end attack strategy (Section 2.4)
            for key_hypothesis in range(256):

                # Computation of the sensitive values set (see Step 2.1 of the key recovery phase depicted in Section 4.2)
                # REMINDER: The following line must be adapted to the targeted dataset
                sensitive_variables = sbox_vectorized(
                    attack_plaintexts[current_subset_index], key_hypothesis
                )

                # Projection of targeted variables into Guilley etal. orthonormal basis [GHMR17] (see Step 2.2 of the key recovery phase depicted in Section 4.2)
                # REMINDER: The following line must be adapted to the targeted dataset
                projection_sensitive_variables = orthonormal_basis_projection(
                    sensitive_variables, max_nb_monomials_interactions, len_basis
                )

                # Computation of the optimal dimensionality reduction given parameters estimated by cVAE-OSM model (see Step 2.3 of the key recovery phase depicted in Section 4.2)
                t_tilde = compute_optimal_dimensionality_reduction(
                    traces_attack_subset[:, indexs_pois],
                    projection_sensitive_variables,
                    psi_layer[:, indexs_pois],
                    variance_encoder[indexs_pois],
                )

                # Computation of the score for the current key hypothesis (see Step 2.4 of the key recovery phase depicted in Section 4.2)
                opt_red[key_hypothesis] = np.sum(t_tilde)

            # Computation of the rank for the current attack
            keys_evol.append(np.where(np.argsort(opt_red) == true_key)[0][0])

        # Computation of the mean rank
        mean_rank[current_nb_traces] = np.mean(keys_evol)

        print("Number of traces :", nb_traces[: current_nb_traces + 1])
        print("Mean rank evolution :", mean_rank[: current_nb_traces + 1])

    return mean_rank
