"""
Once models are trained we run this script so we run monte carlo simulation
to create the gradcam masks but also store the output logits

will create an hdf5 for index # so we can link with the test dataset
100 x length x width x channels size for gradcam images
100 x num_classes for output logits
"""

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential

import tensorflow_datasets as tfds

import numpy as np
import os
import matplotlib.pyplot as plt

def load_dataset_from_h5(name):
    with h5py.File(f"datasets/{name}.h5", "r") as f:
        test_x = np.array(f["test_x"])
        test_y = np.array(f["test_y"])
    return test_x, test_y

def get_train_size_from_h5(name):
    with h5py.File(f"datasets/{name}.h5", "r") as f:
        train_y = f["train_y"]
        return train_y.shape[0]  # number of training samples


def sample_evenly_across_classes(images, labels, num_classes, num_to_pick, seed=42):
    np.random.seed(seed)
    sampled_images = []
    sampled_labels = []

    for cls in range(num_classes):
        # Find indices where the one-hot label has a 1 in column cls
        class_indices = np.where(labels[:, cls] == 1)[0]

        if len(class_indices) < num_to_pick:
            raise ValueError(
                f"Not enough samples for class {cls}. Requested {num_to_pick}, but only {len(class_indices)} available."
            )

        chosen_indices = np.random.choice(class_indices, num_to_pick, replace=False)

        sampled_images.append(images[chosen_indices])
        sampled_labels.append(labels[chosen_indices])

    sampled_images = np.concatenate(sampled_images, axis=0)
    sampled_labels = np.concatenate(sampled_labels, axis=0)

    # Shuffle final result so classes aren't grouped
    perm = np.random.permutation(len(sampled_labels))
    return sampled_images[perm], sampled_labels[perm]




tfd = tfp.distributions
tfpl = tfp.layers




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


def make_gradcam_heatmaps_for_all_classes(img_array, model, num_classes, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # https://keras.io/examples/vision/grad_cam/
    # need to make sure that we store the gradients for each image, then go through each index
    # this is essential since otherwise the gradcam mask wont match with the other probability values since they change
    # for example: if we predict for an image of a 0
    # and we get the logit probabilities of (1,10) and the gradcam mask for class 0
    # then we run the same image but get the mask for class 1, the predicted probabilities have changed
    # So it essential we compute all the gradcams masks using only 1 prediction so that way the masks correspond to the
    # predicted probabilities

    image = tf.convert_to_tensor(img_array[np.newaxis, ...])
    heatmaps = []
    pred = []

    # One forward+backward pass for all classes
    with tf.GradientTape(persistent=True) as tape:
        # Watch the image if necessary
        tape.watch(image)

        last_conv_layer_output, preds = grad_model(image)

        # If it's a TFP output with .logits
        preds = preds.logits


        pred.append(preds.numpy())  # store for later

        # print("preds:")
        # print(preds)

        # having this loop inside here and not predicting again ensure we are able to get gradcam masks for all classes
        # without needing to make addition predictions
        # this ensures each masks matches the probabilities for a given sample
        for class_idx in range(num_classes):
            class_channel = preds[:, class_idx]
            grads = tape.gradient(class_channel, last_conv_layer_output)

            # print("Class chanell:")
            # print(class_channel)

            if grads is None:
                print(f"Warning: grads is None for class {class_idx}")
                heatmaps.append(np.zeros_like(last_conv_layer_output[0, :, :, 0]))
                continue

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # DO NOT overwrite last_conv_layer_output — use a new variable
            conv_output = last_conv_layer_output[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Normalize
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap /= max_val

            #print("heatmap!")
            #print(heatmap)

            heatmaps.append(heatmap.numpy())

        del tape  # free memory

    return heatmaps, pred

def monte_carlo_gradcam_and_preds(model, image, class_index, target_layer, num_passes):
    heatmaps = []
    preds = []

    for _ in range(num_passes):
        heatmap, pred = make_gradcam_heatmaps_for_all_classes(image, model, class_index, target_layer)
        heatmaps.append(heatmap)
        preds.append(pred)

    heatmaps = np.array(heatmaps)
    preds = np.array(preds)  # shape: (num_passes, num_classes)

    # avg_heatmap = np.mean(heatmaps, axis=0)
    return heatmaps, preds

# num_classes = 10
images_from_each_class = 50
times_to_sample = 100

seed = 10
# test_index = 1


datasets = {
    "mnist": {
        "num_classes": 10,
        "gradcam_shape": (10, 10)
    }

    ,
    "fashion": {
        "num_classes": 10,
        "gradcam_shape": (6, 6)
    },
    # "cifar": {
    #     "num_classes": 10,
    #     "gradcam_shape": (10, 10)  # e.g. CIFAR might have RGB conv maps
    # },
    "kmnist": {
        "num_classes": 10,
        "gradcam_shape": (6, 6)
    },
    "emnist": {
        "num_classes": 47,
        "gradcam_shape": (6, 6)
    }
}

for name, config in datasets.items():
    num_classes = config["num_classes"]
    gradcam_shape = config["gradcam_shape"]

    print(f"\nProcessing dataset: {name.upper()}")

    # Load test data
    test_x, test_y = load_dataset_from_h5(name)

    # Sample evenly across classes
    sampled_x, sampled_y = sample_evenly_across_classes(test_x, test_y, num_classes, images_from_each_class, seed=seed)

    # Get dataset shape from sampled data (e.g. (28,28,1) for MNIST)
    dataset_shape = sampled_x.shape[1:]  # (height, width, channels)

    # Load model for this dataset
    train_size = get_train_size_from_h5(name)
    model = build_model(dataset_shape, num_classes, train_size)
    model.load_weights(f"weights/bcnn_{name}_best_weights.h5")

    # Find last Conv2D layer for Grad-CAM
    target_layer_name = None
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name.lower():
            target_layer_name = layer.name
            break
    if target_layer_name is None:
        raise ValueError("No Conv2D layer found in model.")
    # target_layer = model.get_layer(target_layer_name)
    print(f"Target layer for Grad-CAM: {target_layer_name}")

    num_of_images = sampled_x.shape[0]

    # Prepare HDF5 file path
    out_path = f"gradcam_outputs/{name}_gradcam_outputs.h5"

    with h5py.File(out_path, "w") as f:
        # Store inputs and labels
        f.create_dataset("test_x", data=sampled_x, compression="gzip")
        f.create_dataset("test_y", data=sampled_y, compression="gzip")

        heatmap_shape = (num_of_images, times_to_sample, num_classes) + gradcam_shape

        logits_shape = (num_of_images, times_to_sample, num_classes)


        heatmaps_dset = f.create_dataset("heatmaps", shape=heatmap_shape, dtype='float32', compression="gzip")
        logits_dset = f.create_dataset("logits", shape=logits_shape, dtype='float32', compression="gzip")

        for i in range(num_of_images):
            print(f"Processing image {i + 1}/{num_of_images}")

            # Get current input image
            input_image = sampled_x[i]

            # Run Grad-CAM multiple times (MC sampling)
            heatmap, preds = monte_carlo_gradcam_and_preds(
                model, input_image, num_classes, target_layer_name, times_to_sample
            )

            # (samples, 1, 1, num_classes) idk why the extra 1s

            # Each prediction in gradcam is (1,10) so we squeuze out the 1 in the middle
            preds = np.squeeze(preds, axis=(1,2))

            # print("Shapes")
            # print(heatmap.shape)
            # print(preds.shape)

            # Now save safely
            heatmaps_dset[i] = heatmap
            logits_dset[i] = preds

    print("Finished")









#
#
# test_x, test_y = load_dataset_from_h5("mnist")
# num_classes = 10
# num_to_pick = 2
# seed = 10
# test_index = 1
# number_of_passes = 2
#
# sampled_x, sampled_y = sample_evenly_across_classes(test_x, test_y, num_classes, num_to_pick, seed=seed)
#
# image = sampled_x[test_index]
# label = sampled_y[test_index]
# class_index = np.argmax(label)
#
# print(image.shape)
# print(label)
# print(class_index)
#
# model = build_model((28,28,1), num_classes, get_train_size_from_h5("mnist"))
# model.load_weights(f"weights/bcnn_mnist_best_weights.h5")
#
# # Find last Conv2D layer name
# target_layer_name = None
# for layer in reversed(model.layers):
#     if 'conv2d' in layer.name.lower():
#         target_layer_name = layer.name
#         break
#
# target_layer = model.get_layer(target_layer_name)
#
# heatmap, preds = monte_carlo_gradcam_and_preds(
#         model, image, num_classes, target_layer, number_of_passes
#     )
#
#
#
#
#
#
#
#
#
#
#
# num_to_pick = 1  # e.g. 100 samples per class
# times_to_sample = 2
# seed = 42
#
# # This will store the heatmaps for each class, this can use a lot of storage since we are now saving
# # num_to_pick * times_to_sample * num_classes * num_classes
# # will be number of classes times larger and take much longer to compute
# store_all_heatmaps = True

# for name, num_classes in datasets.items():
#     print(f"\nProcessing {name.upper()}...")
#     test_x, test_y = load_dataset_from_h5(name)
#     sampled_x, sampled_y = sample_evenly_across_classes(test_x, test_y, num_classes, num_to_pick, seed=seed)
#
#     class_indices = np.argmax(sampled_y, axis=1)
#     dataset_shape = (sampled_x.shape[1], sampled_x.shape[2], sampled_x.shape[3])
#     train_size = get_train_size_from_h5(name)
#
#     model = build_model(dataset_shape, num_classes, train_size)
#     model.load_weights(f"weights/bcnn_{name}_best_weights.h5")
#
#     # Find last Conv2D layer
#     target_layer_name = next((layer.name for layer in reversed(model.layers) if 'conv2d' in layer.name.lower()), None)
#     target_layer = model.get_layer(target_layer_name)
#     print("Target layer:", target_layer_name)
#
#     # Grad-CAM output shapes
#     num_samples = sampled_x.shape[0]
#     heatmap_class_count = num_classes if store_all_heatmaps else 1
#
#     # Prepare HDF5 file
#     out_path = f"gradcam_outputs/{name}_gradcam_outputs.h5"
#     with h5py.File(out_path, "w") as f:
#         f.create_dataset("test_x", data=sampled_x, compression="gzip")
#         f.create_dataset("test_y", data=sampled_y, compression="gzip")
#
#         heatmap_shape = (num_samples, times_to_sample, heatmap_class_count,
#                          dataset_shape[0], dataset_shape[1], dataset_shape[2])
#         logits_shape = (num_samples, times_to_sample, num_classes)
#
#         heatmaps_dset = f.create_dataset("heatmaps", shape=heatmap_shape, dtype='float32', compression="gzip")
#         logits_dset = f.create_dataset("logits", shape=logits_shape, dtype='float32', compression="gzip")
#
#         for i in range(num_samples):
#             for t in range(times_to_sample):
#                 # Run the model multiple times to sample different dropout paths
#                 input_sample = sampled_x[i]
#                 print(input_sample.shape)
#
#                 preds = model(input_sample, training=True).logits.numpy()
#                 logits_dset[i, t] = preds[0]
#
#                 # Compute Grad-CAM heatmaps
#                 if store_all_heatmaps:
#                     for c in range(num_classes):
#                         heatmap, preds = monte_carlo_gradcam_and_preds(
#                             model, input_sample, class_index=c, target_layer=target_layer, num_passes=times_to_sample
#                         )
#                         heatmaps_dset[i, t, c] = heatmap
#                 else:
#                     actual_class = np.argmax(sampled_y[i])
#                     heatmap, preds = monte_carlo_gradcam_and_preds(
#                         model, input_sample, class_index=c, target_layer=target_layer, num_passes=times_to_sample
#                     )
#                     heatmaps_dset[i, t, 0] = heatmap
#
#     print(f"Saved outputs to {out_path}")





