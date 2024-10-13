import tensorflow as tf
from tensorflow import keras
from keras import backend


class GreaterThanZeroConstraint(keras.constraints.Constraint):
    """
    This class is a custom constraint for kernel weights and inherits from keras Constraint class.
    It forces the weights to be strictly greater than 0  (see Section 5.1.1, paragraph Implementation tricks).
    """

    def __call__(self, w):
        """
        Arguments:
            w: kernel weights

        Returns:
            Weights that are striclty greater than 0.
        """
        epsilon = 1 / 10
        w = w * tf.cast(tf.greater(w, 0.0), backend.floatx())
        w += epsilon * tf.cast(tf.equal(w, 0.0), backend.floatx())
        return w

    def get_config(self):
        """
        This methods enables a proper saving of the model.

        Returns:
            The configuration.

        """
        config = super().get_config()

        return config
