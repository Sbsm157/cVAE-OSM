from math import pi
import tensorflow as tf
from tensorflow import keras
from keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, concatenate, Lambda, Identity

from .Kernel_Weights_Constraints import GreaterThanZeroConstraint

# =====================================================================
#
# ---------- Class implementing custom variance dense layer -----------
#
# =====================================================================


class Dense_Variance(Layer):
    """
    This class ensures that the variance converges to 2*D and shares the same weights as the Dense layer that computes the mean (see Section 5.1.1, paragraph Implementation tricks).
    """

    def __init__(self, original_layer, **kwargs):
        """
        Instantiation of a Dense_Variance Layer. This class inherits from keras Layer class.

        Arguments:
            orginial_layer: Dense layer that computes the mean
            **kwargs: standard Layer keyword arguments
        """
        super().__init__(**kwargs)

        self.original_layer = original_layer

    def call(self, inputs):
        """
        Arguments:
            inputs: multivariate traces T

        Returns:
            The variance of the monovariate traces T̃ which is computed by using
            the same weights as the Dense layer that computes the mean.

        """
        return tf.matmul(inputs, 2 * self.original_layer.weights[0])

    def get_config(self):
        """
        This methods enables a proper saving of the model.

        Returns:
            The configuration in which the arguments are serialized.

        """
        base_config = super().get_config()
        config = {
            "original_layer": keras.saving.serialize_keras_object(self.original_layer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        This methods enables a proper saving of the model.

        Returns:
            The deserialized arguments.

        """
        original_layer_config = config.pop("original_layer")
        original_layer = keras.saving.deserialize_keras_object(original_layer_config)
        return cls(original_layer, **config)


# =====================================================================
#
# ----- Class implementing the sampling (reparametrization trick) -----
#
# =====================================================================


class Sampling(Layer):
    """
    This class corresponds to the reparametrization trick (see Section 3.3).
    It uses (z_mean, z_var) to sample the latent variable z i.e. the optimal dimensionality reduction of
    the input trace T.
    This class is inspired from https://keras.io/examples/generative/vae/.
    """

    def call(self, inputs, dim):
        """
        Arguments:
            inputs: ([z_mean, z_var]) mean and variance of the monovariate trace T̃ (ie
                    optimal reduction dimensionality of T)
            dim: dimension of the input trace T

        Returns:
            A set of dim samples z which follow the multivariate Gaussian distribution N(mu_phi, Sigma_phi).
        """
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.math.sqrt(z_var) * epsilon

    def get_config(self):
        """
        This methods enables a proper saving of the model.

        Returns:
            The configuration.

        """
        config = super().get_config()

        return config


# =====================================================================
#
# ------------------- Class implementing cVAE-OSM ---------------------
#
# =====================================================================


class cVAE(Model):
    """
    This class implements cVAE-OSM (see Section 3.3).
    It is inspired from https://keras.io/examples/generative/vae/.
    """

    def __init__(self, encoder, decoder, **kwargs):
        """
        Instantiation of a cVAE-OSM. This class inherits from keras Model class.

        Arguments:
            encoder: encoder of a cVAE-OSM
            decoder: decoder of a cVAE-OSM
            **kwargs: standard Model keyword arguments
        """

        super(cVAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        # Initialization of the learning metrics
        self.total_loss_tracker = keras.metrics.Mean(name="ELBO_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """
        Returns:
            The learning metrics.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        """
        Computation of the Elbo loss terms during the training process.

        Arguments:
            data: ([input traces, orthonormal basis]) input traces T and the orthonormal bases
                   encoding their associated sensitive variable Y

        Returns:
            The result of each loss, namely the Elbo, reconstruction and kl-divergence
            losses s.t. elbo = reconstruction + kl-divergence losses.
        """
        with tf.GradientTape() as tape:
            # 1) Computation of each term that are needed for the loss computation

            ## 1.0) Getting back the dimension of input traces and the batch_size
            dim = tf.constant(data[0][0].get_shape().as_list()[1], dtype=tf.float32)
            dim_size = tf.shape(data[0][0])[1]
            batch_size = tf.shape(data[0][0])[0]

            ## 1.1) Computation of the mean/variance of the latent space Z and sampling of z ∈ Z done by the encoder
            z_mean, z_var, z_sample = self.encoder([data[0][0], data[0][1]])

            ## 1.2) Getting back from the encoder the weights i.e. the inverse of the variance for each sample of traces
            encoder_weights_inverse_variance_vector = self.encoder.get_layer(
                "z_mean"
            ).weights[0]
            encoder_weights_inverse_variance = tf.tile(
                encoder_weights_inverse_variance_vector, [batch_size, 1]
            )
            encoder_weights_inverse_variance = tf.reshape(
                encoder_weights_inverse_variance, [batch_size, dim_size]
            )
            encoder_weights_variance = tf.math.pow(encoder_weights_inverse_variance, -1)

            ## 1.3) Reconstruction of traces done by the decoder
            reconstruction, decoder_psi_layer = self.decoder(
                [z_sample, data[0][1], encoder_weights_variance]
            )

            # 2) Computation of the reconstruction loss
            reconstruction_loss = (
                tf.math.log(2 * pi * encoder_weights_variance)
                + tf.math.square(data[0][0] - decoder_psi_layer)
                * encoder_weights_inverse_variance
            )
            reconstruction_loss = 0.5 * reconstruction_loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(reconstruction_loss, axis=1)
            )

            # 3) Computation of the KL-divergence loss
            kl_loss = (
                tf.math.log(2 * dim / z_var)
                + ((tf.math.square(z_mean - dim) + z_var) / (2 * dim))
                - 1
            )
            kl_loss = 0.5 * kl_loss
            kl_loss = tf.reduce_mean(kl_loss)

            # 4) Computation of the ELBO loss
            total_loss = reconstruction_loss + kl_loss

        # 5) Computation and application of the gradient descent algorithm
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # 6) Update of the loss values
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "ELBO loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, val_data):
        """
        Computation of the validation loss terms during the training process.

        Arguments:
            val_data: ([input validation traces, orthonormal basis]) input validation traces T and the
                      orthonormal bases encoding their associated sensitive variable Y

        Returns:
            The result of each validation loss, namely validation Elbo, reconstruction and kl-divergence losses
            s.t. elbo = reconstruction + kl-divergence losses.
        """
        # 1) Computation of each term that are needed for the validation loss computation

        ## 1.0) Getting back the dimension of input traces and the batch_size
        dim = tf.constant(val_data[0].get_shape().as_list()[1], dtype=tf.float32)
        dim_size = tf.shape(val_data[0])[1]
        batch_size = tf.shape(val_data[0])[0]

        ## 1.1) Computation of the mean/variance of the latent space Z and sampling of z ∈ Z done by the encoder
        val_z_mean, val_z_var, val_z_sample = self.encoder([val_data[0], val_data[1]])

        ## 1.2) Getting back from the encoder the weights i.e. the inverse of the variance for each sample of traces
        val_encoder_weights_inverse_variance_vector = self.encoder.get_layer(
            "z_mean"
        ).weights[0]
        val_encoder_weights_inverse_variance = tf.tile(
            val_encoder_weights_inverse_variance_vector, [batch_size, 1]
        )
        val_encoder_weights_inverse_variance = tf.reshape(
            val_encoder_weights_inverse_variance, [batch_size, dim_size]
        )
        val_encoder_weights_variance = tf.math.pow(
            val_encoder_weights_inverse_variance, -1
        )

        ## 1.3) Reconstruction of traces done by the decoder
        val_reconstruction, val_decoder_psi_layer = self.decoder(
            [val_z_sample, val_data[1], val_encoder_weights_variance]
        )

        # 2) Computation of the validation reconstruction loss
        val_reconstruction_loss = (
            tf.math.log(2 * pi * val_encoder_weights_variance)
            + tf.math.square(val_data[0] - val_decoder_psi_layer)
            * val_encoder_weights_inverse_variance
        )
        val_reconstruction_loss = 0.5 * val_reconstruction_loss
        val_reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(val_reconstruction_loss, axis=1)
        )

        # 3) Computation of the validation KL-divergence loss
        val_kl_loss = (
            tf.math.log(2 * dim / val_z_var)
            + ((tf.math.square(val_z_mean - dim) + val_z_var) / (2 * dim))
            - 1
        )
        val_kl_loss = 0.5 * val_kl_loss
        val_kl_loss = tf.reduce_mean(val_kl_loss)

        # 4) Computation of the validation ELBO loss
        val_total_loss = val_reconstruction_loss + val_kl_loss

        # 5) Update of the validation loss values
        self.total_loss_tracker.update_state(val_total_loss)
        self.reconstruction_loss_tracker.update_state(val_reconstruction_loss)
        self.kl_loss_tracker.update_state(val_kl_loss)

        return val_total_loss


# =====================================================================
#
# -------------- Functions defining encoder and decoder ---------------
#
# =====================================================================


def define_encoder(input_size, len_basis=256, is_deterministic=True, seed=42):
    """
    Construction of the encoder (see Section 3.3).

    Arguments:
        input_size: dimension of the input traces T
        len_basis: size of the orthonormal basis i.e. the monomial subspace (maximal degree of bit interactions=0 => len_basis=1 /
                maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        is_deterministic: boolean used to produce reproductible results. If is_deterministic is set to True, a value of seed must be specified
        seed: value of seed for reproductible results

    Returns:
        Encoder model.
    """
    if is_deterministic:
        weights_init = tf.keras.initializers.GlorotUniform(seed=seed)
    else:
        weights_init = tf.keras.initializers.GlorotUniform()

    # Input initialization (input traces, orthonormal basis)
    input_shape1 = (input_size,)
    tr_input = Input(shape=input_shape1, name="trace")

    input_shape2 = (len_basis,)
    base_input = Input(shape=input_shape2, name="orthonormal_basis_encoder")

    # Extraction of the leakage model part (psi layer)
    psi_layer = Dense(
        input_size,
        activation=None,
        use_bias=False,
        name="psi_layer_encoder",
        kernel_initializer=weights_init,
    )(base_input)

    # Noise extraction i.e. computation of the numerator of the Optimal dimensionality reduction (see Theorem 2)
    noise_layer = Lambda(
        lambda x: tf.math.square(x[0] - x[1]), name="noise_estimation"
    )([tr_input, psi_layer])

    # Output layers i.e. mean and variance of the optimal dimensionality reduction
    dense_mean = Dense(
        1,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Ones(),
        kernel_constraint=GreaterThanZeroConstraint(),
        name="z_mean",
    )

    z_mean = dense_mean(noise_layer)

    z_var = Dense_Variance(dense_mean, trainable=False, name="z_var")(noise_layer)

    # Reparametrization trick
    z = Sampling()([z_mean, z_var], input_size)

    # Creation of the model
    encoder = Model([tr_input, base_input], [z_mean, z_var, z], name="encoder")

    # encoder.summary()

    return encoder


def define_decoder(input_size, len_basis=256, is_deterministic=True, seed=42):
    """
    Construction of the decoder (see Section 3.3).

    Arguments:
        input_size: dimension of the input traces T
        len_basis: size of the orthonormal basis i.e. the monomial subspace (maximal degree of bit interactions=0 => len_basis=1 /
                maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        is_deterministic: boolean used to produce reproductible results. If is_deterministic is set to True, a value of seed must be specified
        seed: value of seed for reproductible results

    Returns:
        Decoder model.
    """
    if is_deterministic:
        weights_init = tf.keras.initializers.GlorotUniform(seed=seed)
    else:
        weights_init = tf.keras.initializers.GlorotUniform()

    # Input initialization (input latent representation, orthonormal basis, weights of encoder)
    input_shape1 = (input_size,)
    latent_inputs = Input(shape=input_shape1, name="z")  # z: Latent representation

    input_shape2 = (len_basis,)
    base_input = Input(shape=input_shape2, name="orthonormal_basis")

    encoder_variance = Input(shape=input_shape1, name="encoder_variance")

    # Construction of the noise 1/2: normalization of the noise (normalized_noise_layer)
    normalized_noise_layer = Lambda(
        lambda x: (x - input_size)
        / tf.math.sqrt(tf.constant(2 * input_size, dtype="float32")),
        name="Normalization_noise",
    )(latent_inputs)

    # Construction of the noise 2/2 (noise_layer)
    noise_layer = Lambda(lambda x: x[0] * tf.math.sqrt(x[1]))(
        [normalized_noise_layer, encoder_variance]
    )

    # Construction of the psi layer
    psi_layer = Dense(
        input_size,
        activation=None,
        use_bias=False,
        name="psi_layer_decoder",
        kernel_initializer=weights_init,
    )(base_input)

    # Construction of the synthetic traces (T synthetic = psi + N_theta)
    synthetic_trace = Lambda(lambda x: x[0] + x[1], name="synthetic_trace")(
        [noise_layer, psi_layer]
    )

    # Creation of the model
    decoder = Model(
        [latent_inputs, base_input, encoder_variance],
        [synthetic_trace, psi_layer],
        name="decoder",
    )

    # decoder.summary()

    return decoder
