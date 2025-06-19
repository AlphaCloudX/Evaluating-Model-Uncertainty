import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image as pil
import tensorflow as tf
import tensorflow_probability as tfp
import h5py
from scipy.ndimage import zoom

tfd = tfp.distributions
tfpl = tfp.layers

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def get_convolutional_reparameterization_layer(filters, kernel_size, stride, padding, divergence_fn):
    return tfpl.Convolution2DReparameterization(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        activation='relu',
        padding=padding,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence_fn
    )

def get_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1.0),
            reinterpreted_batch_ndims=1))
    ])

def get_posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())

class PrimaryCapsBayesian(tf.keras.layers.Layer):
    def __init__(self, n_caps, dims_caps, kernel_size, strides, padding, divergence_fn, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.n_caps = n_caps
        self.dims_caps = dims_caps
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.divergence_fn = divergence_fn

    def get_config(self):
        config = super().get_config()
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
        x = self.conv2d(inputs)
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        x = tf.reshape(x, [batch_size, height * width * self.n_caps, self.dims_caps])
        return squash(x)

class DenseCapsBayesian(tf.keras.layers.Layer):
    def __init__(self, n_caps, dims_caps, r_iter=3, shared_weights=1, divergence_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.n_caps = n_caps
        self.dims_caps = dims_caps
        self.r_iter = r_iter
        self.shared_weights = shared_weights
        self.divergence_fn = divergence_fn

    def build(self, input_shape):
        self.input_n_caps = input_shape[1]
        self.input_dims_caps = input_shape[2]
        weight_shape = (1, self.input_n_caps // self.shared_weights, self.n_caps, self.dims_caps, self.input_dims_caps)
        self.flat_size = tf.reduce_prod(weight_shape)
        self.weight_shape = weight_shape
        self.posterior = get_posterior(self.flat_size, 0, dtype=self.dtype)
        self.prior = get_prior(self.flat_size, 0, dtype=self.dtype)

    def call(self, inputs):
        dist_q = self.posterior(inputs)
        flat_W = dist_q.sample()
        dist_p = self.prior(inputs)
        if self.divergence_fn is not None:
            kl = self.divergence_fn(dist_q, dist_p, None)
            self.add_loss(kl)
        W_sample = tf.reshape(flat_W, self.weight_shape)
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, -1), 2)
        W_tiled = tf.tile(W_sample, [1, self.shared_weights, 1, 1, 1])
        predictions = tf.matmul(W_tiled, inputs_expanded)
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_caps": self.n_caps,
            "dims_caps": self.dims_caps,
            "r_iter": self.r_iter,
            "shared_weights": self.shared_weights,
            # Note: divergence_fn is a function, not serializable; omit or handle specially
        })
        return config

def epsilon():
    return 1e-7

def compute_vectors_length(vecs, axis=-1):
    return tf.sqrt(tf.reduce_sum(tf.square(vecs), axis) + epsilon())

def margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0.0, 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0.0, y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

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

    outputs = tf.keras.layers.Lambda(compute_vectors_length, name="vec_len")(caps_output)

    return tf.keras.Model(inputs=inputs, outputs=[outputs, routing_weights])

def load_dataset_from_h5(name):
    import h5py
    with h5py.File(f"datasets/{name}.h5", "r") as f:
        return np.array(f["train_x"]), np.array(f["train_y"]), np.array(f["val_x"]), np.array(f["val_y"])

# Load data (e.g. MNIST shape = (28, 28, 1))
train_x, train_y, val_x, val_y = load_dataset_from_h5("mnist")

# Parameters
input_shape = train_x.shape[1:]  # (28,28,1)
num_classes = 10
train_size = train_x.shape[0]

# Build the model
model = build_model(shape=input_shape, num_classes=num_classes, train_size=train_size)

# Load weights
weights_path = "weights/BCapsNet_mnist_best_weights.h5"
model.load_weights(weights_path)

# Select an example input image (batch size 1)
example_img = val_x[:1]

routing_weights_outputs = []
vec_len_outputs = []

# Run model prediction 5 times (Monte Carlo sampling)
for i in range(5):
    vec_len_output, routing_weights = model(example_img, training=False)
    routing_weights_outputs.append(routing_weights)
    vec_len_outputs.append(vec_len_output)
    print(vec_len_output)

# Get layers
primary_caps_layer = model.get_layer("primary_caps")
digit_caps_layer = model.get_layer("digit_caps")

# Create intermediate models to get activations (only once)
primary_caps_model = tf.keras.Model(inputs=model.input, outputs=primary_caps_layer.output)
digit_caps_model = tf.keras.Model(inputs=model.input, outputs=digit_caps_layer.output)

prev_act = primary_caps_model(example_img, training=False)  # shape depends on your model
curr_act = digit_caps_model(example_img, training=False)

# Unpack layers and activations for config
prev_layer_pack = ((primary_caps_layer, None), prev_act)
layer_pack = ((digit_caps_layer, None), curr_act)

((pl, _), prev_act) = prev_layer_pack
((cl, _), curr_act) = layer_pack

pl_conf = pl.get_config()
cl_conf = cl.get_config()

# Calculate feature dimension (assuming square)
feature_dim = int((pl.input_shape[1] - pl_conf["kernel_size"] + 1) / pl_conf["strides"])

dims = [
    feature_dim,
    feature_dim,
    pl_conf["n_caps"],
    cl_conf["n_caps"],
]

# Prepare previous caps lengths and tile
tmp_dims = dims[0:3] + [1] * (len(dims) - 3)
prev_caps_lengths = tf.reshape(compute_vectors_length(prev_act), tmp_dims)
tmp_dims = [1] * (3) + dims[3 : len(dims)]
prev_caps_lengths_tiled = tf.tile(prev_caps_lengths, tmp_dims)

# Prepare current caps lengths and tile
tmp_dims = [1, 1, 1, dims[3]] + [1] * (len(dims) - 4)
curr_caps_lengths = tf.reshape(compute_vectors_length(curr_act[0]), tmp_dims)
tmp_dims = dims[0:3] + [1] + dims[4 : len(dims)]
curr_caps_lengths_tiled = tf.tile(curr_caps_lengths, tmp_dims)

# Compute all_paths_average for each Monte Carlo run (5 total)
all_paths_averages = []

for i in range(5):
    routing_weights_i = routing_weights_outputs[i]  # routing weights from MC run i

    # Reshape and tile routing weights
    tmp_dims = dims[0:4] + [1] * (len(dims) - 4)
    routing_weights_reshape = tf.reshape(routing_weights_i, tmp_dims)
    tmp_dims = [1] * (4) + dims[4 : len(dims)]
    routing_weights_reshape_tiled = tf.tile(routing_weights_reshape, tmp_dims)

    # Compute routing path visualization for this MC run
    tmp = tf.multiply(routing_weights_reshape_tiled, curr_caps_lengths_tiled)
    all_paths_i = tf.multiply(prev_caps_lengths_tiled, tmp)
    all_paths_avg_i = tf.reduce_sum(all_paths_i, axis=2)  # sum over prev caps dim

    all_paths_averages.append(all_paths_avg_i.numpy())  # convert to numpy for plotting

# Prepare input image (28x28 grayscale)
input_img = example_img[0, :, :, 0]

# Function to resize mask to 28x28 (input size)
def resize_mask(mask, new_shape=(28,28)):
    zoom_factors = (new_shape[0] / mask.shape[0], new_shape[1] / mask.shape[1])
    return zoom(mask, zoom_factors, order=1)

# Plotting: 5 rows (MC runs), 11 columns (input + classes 0-9)
fig, axes = plt.subplots(5, 11, figsize=(22, 10))
fig.suptitle('RPV Routing Masks - Monte Carlo Runs', fontsize=16)

for row in range(5):
    for col in range(11):
        ax = axes[row, col]
        ax.axis('off')

        if col == 0:
            # First column: input image repeated on every row
            ax.imshow(input_img, cmap='gray')
            if row == 0:
                ax.set_title("Input Image")
        else:
            class_idx = col - 1
            mask = all_paths_averages[row][:, :, class_idx]
            mask_resized = resize_mask(mask, new_shape=input_img.shape)
            mask_norm = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-6)

            ax.imshow(input_img, cmap='gray')
            ax.imshow(mask_norm, cmap='jet', alpha=0.5)

            if row == 0:
                ax.set_title(f'Class {class_idx}', pad=25)

            # Add class probability text above the mask for each run and class
            prob = vec_len_outputs[row].numpy()[0, class_idx]  # get prob scalar
            ax.text(
                0.5, 1.05,  # x=middle, y=just above axes
                f"{prob:.3f}",
                color='black',
                fontsize=8,
                ha='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

plt.show()
