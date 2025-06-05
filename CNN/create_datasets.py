import numpy as np
import os
import h5py
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

seed = 42
np.random.seed(seed)


def split_data(x_data, y_data, seed=42):
    print("Starting data split...")
    unique_classes = np.unique(y_data)
    print(f"Unique classes found: {unique_classes}")

    train_indices, val_indices, test_indices = [], [], []

    for cls in unique_classes:
        cls_indices = np.where(y_data == cls)[0]
        print(f"Class {cls}: {len(cls_indices)} samples")

        train_idx, temp_idx = train_test_split(
            cls_indices, test_size=0.3, random_state=seed, shuffle=True
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.3333, random_state=seed, shuffle=True
        )

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)} samples")

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

    # Convert to np arrays for indexing
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    print(f"Total train samples: {len(train_indices)}")
    print(f"Total val samples: {len(val_indices)}")
    print(f"Total test samples: {len(test_indices)}")

    return (x_data[train_indices], y_data[train_indices]), \
           (x_data[val_indices], y_data[val_indices]), \
           (x_data[test_indices], y_data[test_indices])

def save_to_h5(dataset_name, splits, num_classes):
    print(f"Saving datasets to datasets/{dataset_name}.h5 ...")
    os.makedirs("datasets", exist_ok=True)
    with h5py.File(f"datasets/{dataset_name}.h5", "w") as f:
        for split, (x, y) in splits.items():
            print(f"Writing split '{split}' with {len(x)} samples")
            f.create_dataset(f"{split}_x", data=x)
            f.create_dataset(f"{split}_y", data=to_categorical(y, num_classes))
    print("Save complete.")


def load_dataset(load_fn, name, num_classes):
    print(f"Processing {name}...")
    x, y = load_fn()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)
    save_to_h5(name, {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test)
    }, num_classes)


def load_mnist():  # 28x28x1
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    return x[..., np.newaxis] / 255., y


def load_fashion_mnist():
    (x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
    return x[..., np.newaxis] / 255., y


def load_kmnist():
    ds_train = tfds.load('kmnist', split='train', as_supervised=True)
    ds_test = tfds.load('kmnist', split='test', as_supervised=True)

    # Convert to numpy arrays
    x_train = []
    y_train = []
    for image, label in tfds.as_numpy(ds_train):
        x_train.append(image)
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = []
    y_test = []
    for image, label in tfds.as_numpy(ds_test):
        x_test.append(image)
        y_test.append(label)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    return x, y


def load_cifar():
    (x, y), _ = tf.keras.datasets.cifar10.load_data()
    return x / 255., y.flatten()


# The url is now 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
import numpy as np
import os
import gzip
import struct

def load_idx_gz(path):
    with gzip.open(path, 'rb') as f:
        magic, = struct.unpack('>I', f.read(4))
        ndim = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def load_emnist(split="balanced"):
    base_path = r"C:\Users\Michael\tensorflow_datasets\downloads\manual"
    prefix = f"emnist-{split}"

    train_images_path = os.path.join(base_path, f"{prefix}-train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(base_path, f"{prefix}-train-labels-idx1-ubyte.gz")

    # Load the gzip IDX files
    x = load_idx_gz(train_images_path).astype(np.float32) / 255.0
    y = load_idx_gz(train_labels_path).astype(np.int32)

    # Add channel dimension
    x = np.expand_dims(x, axis=-1)

    return x, y


if __name__ == "__main__":
    load_dataset(load_mnist, "mnist", 10)
    load_dataset(load_fashion_mnist, "fashion", 10)
    load_dataset(load_cifar, "cifar", 10)
    load_dataset(load_kmnist, "kmnist", 10)
    load_dataset(load_emnist, "emnist", 47)  # 'balanced' split
