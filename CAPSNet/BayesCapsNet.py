import json

import h5py
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, Dropout, Flatten
from keras.optimizers import Adam

from tensorflow.keras.models import Sequential


import numpy as np
import os

# https://goodboychan.github.io/python/coursera/tensorflow_probability/icl/2021/08/26/01-Bayesian-Convolutional-Neural-Network.html
# https://openaccess.thecvf.com/content_ICCV_2019/papers/Patro_U-CAM_Visual_Explanation_Using_Uncertainty_Based_Class_Activation_Maps_ICCV_2019_paper.pdf


tfd = tfp.distributions
tfpl = tfp.layers

seed = 42

# 2. Set NumPy seed
np.random.seed(seed)

# 3. Set TensorFlow seed
tf.random.set_seed(seed)
from tensorflow.keras.layers import InputLayer

def nll(y_true, y_pred):
    """
    This function should return the negative log-likelihood of each sample
    in y_true given the predicted distribution y_pred. If y_true is of shape
    [B, E] and y_pred has batch shape [B] and event_shape [E], the output
    should be a Tensor of shape [B].
    """

    return -y_pred.log_prob(y_true)

def get_convolutional_reparameterization_layer(filters, kernel_size, stride, padding, divergence_fn):
    layer = tfpl.Convolution2DReparameterization(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        activation='relu',
        padding=padding,  # Preserves spatial resolution
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence_fn
    )

    return layer


def get_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return Sequential([
        tfpl.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1.0),
            reinterpreted_batch_ndims=1))
    ])

def get_posterior(kernel_size, bias_size, dtype=None):
    """
    This function should create the posterior distribution as specified above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the posterior distribution.
    """
    n = kernel_size + bias_size
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])

from tensorflow.keras.layers import Layer, Conv2D, Reshape, Lambda

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())

class PrimaryCapsBayesian(Layer):
    def __init__(
        self,
        n_caps,
        dims_caps,
        kernel_size,
        strides,
        padding,
        divergence_fn,
        activation=None,
        **kwargs,
    ):
        super(PrimaryCapsBayesian, self).__init__(**kwargs)
        self.n_caps = n_caps
        self.dims_caps = dims_caps
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.divergence_fn = divergence_fn

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "n_caps": self.n_caps,
            "dims_caps": self.dims_caps,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
        })
        return config

    def build(self, input_shape):
        self.conv2d = tfpl.Convolution2DReparameterization(
            filters=self.n_caps * self.dims_caps,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=self.divergence_fn,
            name="primarycaps_bayesian_conv2d",
        )

    def call(self, inputs):
        x = self.conv2d(inputs)  # Shape: [B, H, W, n_caps * dims_caps]
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        # Reshape to [B, num_capsules, dims_caps]
        x = tf.reshape(x, [batch_size, height * width * self.n_caps, self.dims_caps])
        return squash(x)



class DenseCapsBayesian(tf.keras.layers.Layer):
    def __init__(
        self,
        n_caps,
        dims_caps,
        r_iter=3,
        shared_weights=1,
        divergence_fn=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_caps = n_caps
        self.dims_caps = dims_caps
        self.r_iter = r_iter
        self.shared_weights = shared_weights
        self.divergence_fn = divergence_fn

    def build(self, input_shape):
        assert len(input_shape) == 3, "Input shape must be [batch, input_n_caps, input_dims_caps]"
        self.input_n_caps = input_shape[1]
        self.input_dims_caps = input_shape[2]

        # shape of the transformation matrix W that each capsule uses to project ot next capsule layer
        weight_shape = (
            1,
            self.input_n_caps // self.shared_weights,
            self.n_caps,
            self.dims_caps,
            self.input_dims_caps
        )
        # This computes the number of parameters the capsule layer will be using
        self.flat_size = tf.reduce_prod(weight_shape)


        self.weight_shape = weight_shape

        # Create posterior and prior distributions for the number of parameters in the model
        # Sample weights during training
        self.posterior = get_posterior(self.flat_size, 0, dtype=self.dtype)

        # create prior distribution over weights
        # acts like regulizer encouraging weights to stay near 0
        self.prior = get_prior(self.flat_size, 0, dtype=self.dtype)

        self.built = True

    def call(self, inputs):
        # Sample from posterior
        dist_q = self.posterior(inputs)
        flat_W = dist_q.sample()  # draw a sample explicitly
        dist_p = self.prior(inputs)

        # Add KL divergence loss
        if self.divergence_fn is not None:
            kl = self.divergence_fn(dist_q, dist_p, None)
            self.add_loss(kl)

        # resample the sampled posterior to the shape of weight matrix
        W_sample = tf.reshape(flat_W, self.weight_shape)

        # Expand input dims and tile W
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, -1), 2)

        # Use the Sampled Weights Here!!!
        W_tiled = tf.tile(W_sample, [1, self.shared_weights, 1, 1, 1])

        predictions = tf.matmul(W_tiled, inputs_expanded)  # shape: [B, input_n_caps, n_caps, dims_caps, 1]

        # === DYNAMIC ROUTING ===
        raw_weights = tf.zeros([tf.shape(inputs)[0], self.input_n_caps, self.n_caps, 1, 1])

        for i in range(self.r_iter):
            routing_weights = tf.nn.softmax(raw_weights, axis=2)


            outputs = tf.reduce_sum(routing_weights * predictions, axis=1, keepdims=True)
            outputs = squash(outputs, axis=-2)

            if i < self.r_iter - 1:
                outputs_tiled = tf.tile(outputs, [1, self.input_n_caps, 1, 1, 1])
                agreement = tf.matmul(predictions, outputs_tiled, transpose_a=True)
                raw_weights += agreement

        return tf.squeeze(outputs, axis=[1, -1]), routing_weights

def epsilon():
    return 1e-7  # or tf.keras.backend.epsilon()

def compute_vectors_length(vecs, axis=-1):
    return tf.sqrt(tf.reduce_sum(tf.square(vecs), axis) + epsilon())

def margin_loss(y_true, y_pred):
    """Local function for margin loss, Eq.(4).

    When y_true[i, :] contains more than one `1`, this loss should work too (not tested).

    Args:
        y_true: Correct labels one-hot encoded (batch_size, n_classes)
        y_pred: Output of the DigitCaps Layer (batch_size, n_classes)
    Returns:
        A scalar loss value.
    """
    # Calculate loss
    L = y_true * tf.square(tf.maximum(0.0, 0.9 - y_pred)) + 0.5 * (
            1 - y_true
    ) * tf.square(tf.maximum(0.0, y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))

def build_model(shape, num_classes, train_size):
    divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / train_size
    inputs = tf.keras.Input(shape=shape)

    x = get_convolutional_reparameterization_layer(16, (9, 9), (1, 1), 'valid', divergence_fn)(inputs)

    x = PrimaryCapsBayesian(n_caps=32, dims_caps=8, kernel_size=9, strides=2,
                            padding='valid', divergence_fn=divergence_fn,
                            activation='relu', name="primary_caps")(x)

    caps_output, routing_weights = DenseCapsBayesian(n_caps=num_classes, dims_caps=16,
                                                     r_iter=3, divergence_fn=divergence_fn,
                                                     name="digit_caps")(x)

    outputs = Lambda(compute_vectors_length, name="vec_len")(caps_output)

    # Probibalistic Outputs
    # Output = capsule lengths (classification probs)
    # logits = tf.norm(caps_output, axis=-1)

    # Output distribution over classes
    # Taking distribution over an array of shape of (,10) where probability entity exists dont sum to 1?
    # So this may break potentially??
    # This doesnt decrease loss for whatever reason
    # outputs = tfpl.OneHotCategorical(
    #     event_size=num_classes,
    #     convert_to_tensor_fn=tfd.Distribution.mode  # or .mean if preferred
    # )(logits)

    # Now return model with multiple outputs if you want both
    return tf.keras.Model(inputs=inputs, outputs=[outputs, routing_weights])


def train_bayes_capsnet(model, data, epochs=10, batch_size=16, lr=1e-3, dataset_name="mnist"):
    (x_train, y_train), (x_test, y_test) = data

    lr_decay_mul = 0.9  # decay factor

    # === Define callbacks ===
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"weights/BCapsNet_{dataset_name}_best_weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lr * (lr_decay_mul ** epoch)
    )

    callbacks_list = [lr_scheduler, early_stop, checkpoint]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=[margin_loss, None],
        metrics=['accuracy']
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
    )

    # === Save training history ===
    os.makedirs("model_history", exist_ok=True)
    with open(f"model_history/{dataset_name}_bcapsnet_history.json", "w") as f:
        json.dump(
            {k: [float(vv) for vv in v] for k, v in history.history.items()},
            f,
            indent=2
        )

    return history


def load_dataset_from_h5(name):
    with h5py.File(f"datasets/{name}.h5", "r") as f:
        train_x = np.array(f["train_x"])
        train_y = np.array(f["train_y"])  # one-hot already
        val_x = np.array(f["val_x"])
        val_y = np.array(f["val_y"])  # one-hot already
    return train_x, train_y, val_x, val_y


if __name__ == "__main__":
    # Load dataset
    dataset_name = "mnist"
    x_train, y_train, x_test, y_test = load_dataset_from_h5(dataset_name)

    # Build your model (assumes build_model is defined elsewhere)
    model = build_model(shape=(28, 28, 1), num_classes=10, train_size=x_train.shape[0])
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    # Train and track weights and history
    train_bayes_capsnet(
        model,
        ((x_train, y_train), (x_test, y_test)),
        batch_size=16,
        epochs=2,
        dataset_name=dataset_name
    )