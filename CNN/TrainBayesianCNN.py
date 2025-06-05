import json

import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, Dropout, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

import tensorflow_datasets as tfds

import numpy as np
import os
import matplotlib.pyplot as plt

# https://goodboychan.github.io/python/coursera/tensorflow_probability/icl/2021/08/26/01-Bayesian-Convolutional-Neural-Network.html
# https://openaccess.thecvf.com/content_ICCV_2019/papers/Patro_U-CAM_Visual_Explanation_Using_Uncertainty_Based_Class_Activation_Maps_ICCV_2019_paper.pdf


tfd = tfp.distributions
tfpl = tfp.layers

seed = 42

# 2. Set NumPy seed
np.random.seed(seed)

# 3. Set TensorFlow seed
tf.random.set_seed(seed)


# def load_mnist():
#     (x, y), _ = tf.keras.datasets.mnist.load_data()
#     x = (x / 255.).astype(np.float32)[..., np.newaxis]
#     y = y.astype(np.float32)
#
#     return x, y
#
#
# def load_cifar():
#     (x, y), _ = tf.keras.datasets.cifar10.load_data()
#     x = (x / 255.).astype(np.float32)
#     y = y.astype(np.float32)
#
#     return (x, y)
#
#
# def load_fashion_mnist():
#     (x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
#     x = (x / 255.).astype(np.float32)[..., np.newaxis]
#     y = y.astype(np.float32)
#     return (x, y)
#
#
# def load_kmnist():
#     (x, y), _ = tf.keras.datasets.kmnist.load_data()
#     x = (x / 255.).astype(np.float32)[..., np.newaxis]  # Add channel dim
#     y = y.astype(np.float32)
#     return (x, y)
#
#
# def load_emnist(split='balanced'):
#     ds = tfds.load(f'emnist/{split}', split='train', as_supervised=True)
#     images, labels = [], []
#
#     for img, label in tfds.as_numpy(ds):
#         images.append(img)
#         labels.append(label)
#
#     x = np.array(images, dtype=np.float32) / 255.
#     x = x[..., np.newaxis]  # Add channel dimension
#     y = np.array(labels, dtype=np.float32)
#
#     return x, y
#
#
# def split_data(x_data, y_data, seed):
#     # Step 1: Find unique classes in y_data
#     unique_classes = np.unique(y_data)
#
#     print(unique_classes)
#
#     # Step 2: Store indices for each class
#     class_indices = {cls: np.where(y_data == cls)[0] for cls in unique_classes}
#
#     # Step 3: Split indices into 70% train, 20% val, 10% test
#     train_indices = []
#     val_indices = []
#     test_indices = []
#
#     for cls, indices in class_indices.items():
#         # Split class indices into train, val, test (70%, 20%, 10%)
#         train, temp = train_test_split(indices, train_size=0.7, random_state=seed)
#         val, test = train_test_split(temp, test_size=0.5, random_state=seed)
#
#         train_indices.extend(train)
#         val_indices.extend(val)
#         test_indices.extend(test)
#
#     # Step 4: Organize x_data and y_data into train, val, test
#     x_train = x_data[train_indices]
#     y_train = y_data[train_indices]
#     x_val = x_data[val_indices]
#     y_val = y_data[val_indices]
#     x_test = x_data[test_indices]
#     y_test = y_data[test_indices]
#
#     # Step 5: Return the datasets
#     return (x_train, y_train), (x_val, y_val), (x_test, y_test)
#
#
# def inspect_images(data, num_images, cmap):
#     fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(2 * num_images, 2))
#     for i in range(num_images):
#         ax[i].imshow(data[i], cmap=cmap)
#         ax[i].axis('off')
#     plt.show()
#
#
# # Load MNIST
# x_mnist, y_mnist = load_mnist()
#
# (train_x_mnist, train_y_mnist), (val_x_mnist, val_y_mnist), (test_x_mnist, test_y_mnist) = split_data(x_mnist, y_mnist,
#                                                                                                       seed)
# inspect_images(train_x_mnist, 8, 'gray')
# mnist_classes = 10
#
# # Load Cifar
# x_cifar, y_cifar = load_cifar()
# (train_x_cifar, train_y_cifar), (val_x_cifar, val_y_cifar), (test_x_cifar, test_y_cifar) = split_data(x_cifar, y_cifar,
#                                                                                                       seed)
#
# inspect_images(train_x_cifar, 8, None)
# cifar_classes = 10
#
# # load fashion mnist
# x_fashion, y_fashion = load_fashion_mnist()
#
# (train_x_fashion, train_y_fashion), (val_x_fashion, val_y_fashion), (test_x_fashion, test_y_fashion) = split_data(
#     x_fashion, y_fashion, seed)
#
# inspect_images(train_x_fashion, 8, 'gray')
# fashion_classes = 10
#
# # Load KMNIST
# x_kmnist, y_kmnist = load_kmnist()
# (train_x_kmnist, train_y_kmnist), (val_x_kmnist, val_y_kmnist), (test_x_kmnist, test_y_kmnist) = split_data(x_kmnist,
#                                                                                                             y_kmnist,
#                                                                                                             seed)
#
# inspect_images(train_x_kmnist, 8, 'gray')
# kmnist_classes = 10
#
# # Load EMNIST
# x_emnist, y_emnist = load_emnist(split='balanced')  # You can also try 'letters', 'digits', etc.
# (train_x_emnist, train_y_emnist), (val_x_emnist, val_y_emnist), (test_x_emnist, test_y_emnist) = split_data(x_emnist,
#                                                                                                             y_emnist,
#                                                                                                             seed)
#
# inspect_images(train_x_emnist, 8, 'gray')
# emnist_classes = len(np.unique(y_emnist))  # Should be 47 for 'balanced', 26 for 'letters'
#
# # One-hot encode labels
# train_y_mnist_oh = tf.keras.utils.to_categorical(train_y_mnist, mnist_classes)
# val_y_mnist_oh = tf.keras.utils.to_categorical(val_y_mnist, mnist_classes)
#
# train_y_cifar_oh = tf.keras.utils.to_categorical(train_y_cifar, cifar_classes)
# val_y_cifar_oh = tf.keras.utils.to_categorical(val_y_cifar, cifar_classes)
#
# train_y_fashion_oh = tf.keras.utils.to_categorical(train_y_fashion, fashion_classes)
# val_y_fashion_oh = tf.keras.utils.to_categorical(val_y_fashion, fashion_classes)
#
# train_y_kmnist_oh = tf.keras.utils.to_categorical(train_y_kmnist, kmnist_classes)
# val_y_kmnist_oh = tf.keras.utils.to_categorical(val_y_kmnist, kmnist_classes)
#
# train_y_emnist_oh = tf.keras.utils.to_categorical(train_y_emnist, emnist_classes)
# val_y_emnist_oh = tf.keras.utils.to_categorical(val_y_emnist, emnist_classes)


def nll(y_true, y_pred):
    """
    This function should return the negative log-likelihood of each sample
    in y_true given the predicted distribution y_pred. If y_true is of shape
    [B, E] and y_pred has batch shape [B] and event_shape [E], the output
    should be a Tensor of shape [B].
    """

    return -y_pred.log_prob(y_true)


# def get_convolutional_reparameterization_layer(input_shape, divergence_fn):
#     """
#     This function should create an instance of a Convolution2DReparameterization
#     layer according to the above specification.
#     The function takes the input_shape and divergence_fn as arguments, which should
#     be used to define the layer.
#     Your function should then return the layer instance.
#     """
#
#     layer = tfpl.Convolution2DReparameterization(
#         input_shape=input_shape, filters=8, kernel_size=(5, 5),
#         activation='relu', padding='VALID',
#         kernel_prior_fn=tfpl.default_multivariate_normal_fn,
#         kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
#         kernel_divergence_fn=divergence_fn,
#         bias_prior_fn=tfpl.default_multivariate_normal_fn,
#         bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
#         bias_divergence_fn=divergence_fn
#     )
#     return layer


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

# def normal_prior(event_shape, dtype):
#     return tfd.Independent(tfd.Normal(
#         loc=tf.zeros(event_shape, dtype=dtype),
#         scale=0.1 * tf.ones(event_shape, dtype=dtype)  # Tighter prior
#     ), reinterpreted_batch_ndims=1)
#
# def get_prior(kernel_size, bias_size, dtype=None):
#     """
#     This function should create the prior distribution, consisting of the
#     "spike and slab" distribution that is described above.
#     The distribution should be created using the kernel_size, bias_size and dtype
#     function arguments above.
#     The function should then return a callable, that returns the prior distribution.
#     """
#     n = kernel_size + bias_size
#     prior_model = Sequential([tfpl.DistributionLambda(lambda t: normal_prior(n, dtype))]) # modify from spike and slab to a normal std 0.1 and mean = 0
#     return prior_model


def spike_and_slab(event_shape, dtype):
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=1.0 * tf.ones(event_shape, dtype=dtype)),
                reinterpreted_batch_ndims=1),
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=10.0 * tf.ones(event_shape, dtype=dtype)),
                reinterpreted_batch_ndims=1)],
        name='spike_and_slab')
    return distribution

def get_prior(kernel_size, bias_size, dtype=None):
    """
    This function should create the prior distribution, consisting of the
    "spike and slab" distribution that is described above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the prior distribution.
    """
    n = kernel_size + bias_size
    prior_model = Sequential([tfpl.DistributionLambda(lambda t: spike_and_slab(n, dtype))])
    return prior_model
#
#
# # x_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]
# # plt.plot(x_plot, tfd.Normal(loc=0, scale=1).prob(x_plot).numpy(), label='unit normal', linestyle='--')
# # plt.plot(x_plot, spike_and_slab(1, dtype=tf.float32).prob(x_plot).numpy(), label='spike and slab')
# # plt.xlabel('x')
# # plt.ylabel('Density')
# # plt.legend()
# # plt.show()
#
#



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


def get_dense_variational_layer(units, prior_fn, posterior_fn, kl_weight):
    """
    This function should create an instance of a DenseVariational layer according
    to the above specification.
    The function takes the prior_fn, posterior_fn and kl_weight as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.DenseVariational(
        units=units, make_posterior_fn=posterior_fn, make_prior_fn=prior_fn, kl_weight=kl_weight
    )


# convolutional_reparameterization_layer = get_convolutional_reparameterization_layer(
#     input_shape=(28, 28, 1), divergence_fn=divergence_fn
# )

# convolutional_reparameterization_layer = get_convolutional_reparameterization_layer(
#     filters=(8,8),
#     kernel_size=(1, 1),
#     divergence_fn=divergence_fn)


from tensorflow.keras.layers import InputLayer


def build_model(shape, num_classes, train_size):
    divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / train_size

    dense_variational_layer = get_dense_variational_layer(num_classes,
                                                          get_prior, get_posterior, kl_weight=1 / train_size
                                                          )

    # Same will pad -> this causes the offsets in the gradcam map
    # Valid will simply drop values, prevents need for pooling and allows a slightly smaller final output layer

    model = Sequential([
        InputLayer(input_shape=shape),

        # LeNet Conv1: 6 filters, 5x5, stride=1, padding='SAME'
        get_convolutional_reparameterization_layer(6, (5, 5), (1, 1), 'SAME', divergence_fn),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        # LeNet Conv2: 16 filters, 5x5, stride=1, padding='VALID'
        get_convolutional_reparameterization_layer(16, (5, 5), (1, 1), 'VALID', divergence_fn),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),

        # Dense1: 120 units
        tfp.layers.DenseVariational(
            units=120,
            make_prior_fn=get_prior,
            make_posterior_fn=get_posterior,
            kl_weight=1.0 / train_size,
            activation='relu'
        ),
        BatchNormalization(),

        # Dense2: 84 units
        tfp.layers.DenseVariational(
            units=84,
            make_prior_fn=get_prior,
            make_posterior_fn=get_posterior,
            kl_weight=1.0 / train_size,
            activation='relu'
        ),
        BatchNormalization(),

        # Output layer with Bayesian Dense
        get_dense_variational_layer(num_classes, get_prior, get_posterior, kl_weight=1.0 / train_size),
        tfp.layers.OneHotCategorical(num_classes)
    ])

    return model


def train_bayesian_model(dataset_name, train_x, train_y, val_x, val_y, build_model_fn, loss_fn, lr, batch_size,
                         n_epochs, num_classes):
    """
    Build, compile, and train a Bayesian CNN model on the given dataset.

    Args:
        dataset_name (str): Used for saving weights (e.g., "cifar", "fashion_mnist").
        train_x (np.ndarray): Training images.
        train_y (np.ndarray): One-hot encoded training labels.
        val_x (np.ndarray): Validation images.
        val_y (np.ndarray): One-hot encoded validation labels.
        build_model_fn (callable): Function to build the model. Should accept (input_shape, num_classes, train_size).
        loss_fn (callable): Loss function to use (e.g., `nll`).
        lr (float): Learning rate for optimizer.
        batch_size (int): Batch size.
        n_epochs (int): Number of epochs.
        num_classes (int): Number of output classes.

    Returns:
        model: Trained model.
        history: Training history object.
    """
    input_shape = train_x.shape[1:]
    train_size = train_x.shape[0]

    model = build_model_fn(input_shape, num_classes, train_size)
    model.compile(
        loss=loss_fn,
        optimizer=Adam(lr),
        metrics=["accuracy"],
        experimental_run_tf_function=False  # only needed for TF Probability layers
    )

    model.summary()

    os.makedirs("weights", exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"weights/bcnn_{dataset_name}_best_weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    history = model.fit(
        train_x, train_y,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=1,
        validation_data=(val_x, val_y),
        callbacks=[early_stop, checkpoint]
    )

    return model, history


batch_size = 32
n_epochs = 500
lr = 0.001


import h5py
import numpy as np

def load_dataset_from_h5(name):
    with h5py.File(f"datasets/{name}.h5", "r") as f:
        train_x = np.array(f["train_x"])
        train_y = np.array(f["train_y"])  # one-hot already
        val_x = np.array(f["val_x"])
        val_y = np.array(f["val_y"])      # one-hot already
    return train_x, train_y, val_x, val_y

datasets = {
    "mnist": 10,
    "fashion": 10, # commenting this one out for now becuase cifar is 32x32, not 28x28 so the flatten layer will yield a different output size
    # "cifar": 10,
    "kmnist": 10,
    "emnist": 47
}

for name, num_classes in datasets.items():
    print(f"Training on {name.upper()}...")
    train_x, train_y, val_x, val_y = load_dataset_from_h5(name)

    model, history = train_bayesian_model(
        dataset_name=name,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        build_model_fn=build_model,
        loss_fn=nll,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        num_classes=num_classes
    )

    with open(f"model_history/{name}_history.json", "w") as f:
        json.dump(history.history, f)


# for dataset in range(5):
#     if dataset == 0:
#         print("Training on MNIST...")
#         bcnn_model_mnist, hist_mnist = train_bayesian_model(
#             dataset_name="mnist",
#             train_x=train_x_mnist,
#             train_y=train_y_mnist_oh,
#             val_x=val_x_mnist,
#             val_y=val_y_mnist_oh,
#             build_model_fn=build_model,
#             loss_fn=nll,
#             lr=lr,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             num_classes=mnist_classes
#         )
#     elif dataset == 1:
#         print("Training on Fashion-MNIST...")
#         bcnn_model_fashion, hist_fashion = train_bayesian_model(
#             dataset_name="fashion",
#             train_x=train_x_fashion,
#             train_y=train_y_fashion_oh,
#             val_x=val_x_fashion,
#             val_y=val_y_fashion_oh,
#             build_model_fn=build_model,
#             loss_fn=nll,
#             lr=lr,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             num_classes=fashion_classes
#         )
#     elif dataset == 2:
#         print("Training on CIFAR-10...")
#         bcnn_model_cifar, hist_cifar = train_bayesian_model(
#             dataset_name="cifar",
#             train_x=train_x_cifar,
#             train_y=train_y_cifar_oh,
#             val_x=val_x_cifar,
#             val_y=val_y_cifar_oh,
#             build_model_fn=build_model,
#             loss_fn=nll,
#             lr=lr,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             num_classes=cifar_classes
#         )
#     elif dataset == 3:
#         print("Training on KMNIST...")
#         bcnn_model_kmnist, hist_kmnist = train_bayesian_model(
#             dataset_name="kmnist",
#             train_x=train_x_kmnist,
#             train_y=train_y_kmnist_oh,
#             val_x=val_x_kmnist,
#             val_y=val_y_kmnist_oh,
#             build_model_fn=build_model,
#             loss_fn=nll,
#             lr=lr,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             num_classes=kmnist_classes
#         )
#     elif dataset == 4:
#         print("Training on EMNIST...")
#         bcnn_model_emnist, hist_emnist = train_bayesian_model(
#             dataset_name="emnist",
#             train_x=train_x_emnist,
#             train_y=train_y_emnist_oh,
#             val_x=val_x_emnist,
#             val_y=val_y_emnist_oh,
#             build_model_fn=build_model,
#             loss_fn=nll,
#             lr=lr,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             num_classes=emnist_classes
#         )
