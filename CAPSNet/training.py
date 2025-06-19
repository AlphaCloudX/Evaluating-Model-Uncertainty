"""
Capsule Network training processes implementation.
Author: Antonio Strippoli
Original Work: Xifeng Guo (https://github.com/XifengGuo/CapsNet-Keras)
"""
import json
import os

import h5py
import numpy as np

# Import TensorFlow & Keras
import tensorflow as tf
from keras import Model, Sequential, Input
from keras.datasets import mnist
from keras.layers import Dense
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.utils import to_categorical

from capslayers import *


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


class WeightsSaver(callbacks.Callback):
    """
    Callback that saves weights after every `save_freq` batches at `save_dir` directory.
    """

    def __init__(self, save_dir, save_freq):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        if batch % self.save_freq == 0:
            # Save model current state for later visualization
            save_name = os.path.join(self.save_dir, f"{self.epoch}-{batch}.h5")
            self.model.save_weights(save_name)


def CapsuleNet(input_shape, n_class, name="CapsuleNetwork"):
    """Capsule Network model implementation, used for MNIST dataset training.

    The model has been adapted from
    the [official paper](https://arxiv.org/abs/1710.09829).

    Arguments:
        input_shape: 3-Dimensional data shape (width, height, channels).
        n_class: Number of classes.
    """
    # --- Encoder ---
    x = Input(shape=input_shape, dtype=tf.float32)

    # Layer 1: ReLU Convolutional Layer
    conv1 = Conv2D(
        filters=256,
        kernel_size=9,
        strides=1,
        padding="valid",
        activation="relu",
        name="conv1",
    )(x)

    # Layer 2: PrimaryCaps Layer
    primary_caps = PrimaryCaps(
        n_caps=32,
        dims_caps=8,
        kernel_size=9,
        strides=2,
        padding="valid",
        activation="relu",
        name="primary_caps",
    )(conv1)

    # Layer 3: DigitCaps Layer: since routing it is computed only
    # between two consecutive capsule layers, it only happens here
    digit_caps, _ = DenseCaps(n_caps=n_class, dims_caps=16, name="digit_caps")(
        primary_caps
    )  # [0]

    # Layer 4: A convenience layer to calculate vectors' length
    vec_len = Lambda(compute_vectors_length, name="vec_len")(digit_caps)

    # --- Decoder ---
    y = Input(shape=(n_class,))

    # Layer 1: A convenience layer to compute the masked capsules' output
    masked = Lambda(mask, name="masked")(
        digit_caps
    )  # Mask using the capsule with maximal length. For prediction

    masked_by_y = Lambda(mask, name="masked_by_y")(
        [digit_caps, y]
    )  # The true label is used to mask the output of capsule layer. For training

    # Layer 2-4: Three Dense layer for the image reconstruction
    decoder = Sequential(name="decoder")
    decoder.add(Dense(512, activation="relu", input_dim=16 * n_class, name="dense_1"))
    decoder.add(Dense(1024, activation="relu", name="dense_2"))
    decoder.add(
        Dense(tf.math.reduce_prod(input_shape), activation="sigmoid", name="dense_3")
    )

    # Layer 5: Reshape the output as the image provided in input
    decoder.add(Reshape(target_shape=input_shape, name="img_reconstructed"))

    # Models for training and evaluation (prediction)
    train_model = Model(
        inputs=[x, y], outputs=[vec_len, decoder(masked_by_y)], name=f"{name}_training"
    )
    eval_model = Model(inputs=x, outputs=[vec_len, decoder(masked)], name=name)

    return train_model, eval_model


def train(
        model,
        data,
        epoch=10,
        batch_size=100,
        lr=0.001,
        lr_decay_mul=0.9,
        lam_recon=0.392,
        dataset_name="blankName"
):
    """Train a given Capsule Network model.

    Args:
        model: The CapsuleNet model to train.
        data: The dataset that you want to train: ((x_train, y_train), (x_test, y_test)).
        epochs: Number of epochs for the training.
        batch_size: Size of the batch used for the training.
        lr: Initial learning rate value.
        lr_decay_mul: The value multiplied by lr at each epoch. Set a larger value for larger epochs.
        lam_recon: The coefficient for the loss of decoder (if present).
        save_dir: Directory that will contain the logs of the training. `None` if you don't want to save the logs.
        weights_save_dir: Directory that will contain the weights saved. `None` if you don't want to save the weights.
        save_freq: The number of batches after which weights are saved.
        :param dataset_name: name of dataset being tested
    Returns:
        The trained model.

    """
    # Unpack data
    (x_train, y_train), (x_test, y_test) = data

    # Understand if the model uses the decoder or not
    n_output = len(model.outputs)

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        loss=[margin_loss, "mse"] if n_output == 2 else [margin_loss],
        loss_weights=[1.0, lam_recon] if n_output == 2 else [1.0],
        metrics=["accuracy"],
    )

    model.summary()

    # Define a callback to reduce learning rate
    # === Define callbacks ===
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    checkpoint = callbacks.ModelCheckpoint(
        f"weights/CapsNet_{dataset_name}_best_weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: lr * (lr_decay_mul ** epoch)
    )

    cbacks = [early_stop, checkpoint, lr_decay]

    # Simple training without data augmentation
    history = model.fit(
        x=(x_train, y_train) if n_output == 2 else x_train,
        y=(y_train, x_train) if n_output == 2 else y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=((x_test, y_test), (y_test, x_test))
        if n_output == 2
        else (x_test, y_test),
        callbacks=cbacks,
    )

    # === Save training history ===
    with open(f"model_history/{dataset_name}_history.json", "w") as f:
        json.dump(
            {k: [float(vv) for vv in v] for k, v in history.history.items()},
            f,
            indent=2  # Optional: makes it human-readable
        )

    return model


# def test(model, data):
#     """
#     Calculate accuracy of the model on the test set.
#     """
#     x_test, y_test = data
#     y_pred, x_recon = model.predict(x_test, batch_size=100)
#     print(
#         "Test acc:",
#         np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0],
#     )

# def prepare_mnist():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
#     x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
#
#     y_train = to_categorical(y_train.astype('int32'))
#     y_test = to_categorical(y_test.astype('int32'))
#
#     return x_train, y_train, x_test, y_test

def load_dataset_from_h5(name):
    with h5py.File(f"datasets/{name}.h5", "r") as f:
        train_x = np.array(f["train_x"])
        train_y = np.array(f["train_y"])  # one-hot already
        val_x = np.array(f["val_x"])
        val_y = np.array(f["val_y"])  # one-hot already
    return train_x, train_y, val_x, val_y


if __name__ == "__main__":
    # Check existance of directories and delete them
    # (do not change the names or visualizer will not recognize them anymore)

    weights_save_dir = "weights"

    # Load dataset
    dataset = "mnist"
    x_train, y_train, x_test, y_test = load_dataset_from_h5(dataset)  # dataset

    # Set model args
    model_params = {
        "input_shape": x_train.shape[1:],
        "n_class": y_train.shape[1],
        "name": os.path.basename(os.path.dirname(__file__)),
    }

    # Instantiate Capsule Network Model
    model, eval_model = CapsuleNet(**model_params)

    # Show a complete summary
    model.summary()

    # Train
    model = train(
        model=model,
        data=((x_train, y_train), (x_test, y_test)),
        epoch=1,
        batch_size=100,
        lr=0.001,
        lr_decay_mul=0.9,
        lam_recon=0.392,
        dataset_name=dataset,
    )
