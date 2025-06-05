import numpy as np
import h5py
import tensorflow as tf


def load_logits_from_h5(filepath):
    """
    Load logits from an HDF5 file. Expected shape: (num_images, T, num_classes)
    """
    with h5py.File(filepath, "r") as f:
        logits = f["logits"][:]
    return logits  # shape: (N, T, C)


def compute_uncertainties(logits, use_softplus=False):
    """
    Compute epistemic and aleatoric uncertainties from logits.

    Args:
        logits: numpy array or tensor of shape (N, T, C)
            - N: number of data points
            - T: number of stochastic forward passes (e.g., MC Dropout or Bayesian samples)
            - C: number of classes
        use_softplus: whether to use Softplus normalization instead of Softmax

    Returns:
        pred_mean: (N, C) - Predictive mean (expected class probabilities)
        epistemic_uncertainty: (N, C) - Variance from model uncertainty (epistemic)
        aleatoric_uncertainty: (N, C) - Variance from data noise (aleatoric)
    """
    N, T, C = logits.shape

    # Convert logits to probabilities using either Softmax or Softplus (normalized)
    # Equation (12): p̂_t = softmax(f_{w_t}(x*))
    if use_softplus:
        probs = tf.nn.softplus(logits)
        probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    else:
        probs = tf.nn.softmax(logits, axis=-1)

    # Convert TensorFlow tensor to NumPy array for NumPy operations
    probs_np = probs.numpy()  # shape: (N, T, C)

    # Predictive mean: Equation (12): p̄ = (1/T) ∑_t p̂_t
    p_bar = np.mean(probs_np, axis=1)  # shape: (N, C)

    # Epistemic uncertainty (model uncertainty):
    # Equation (14), second term:
    # (1/T) ∑_t (p̂_t - p̄)(p̂_t - p̄)^T
    temp = probs_np - np.expand_dims(p_bar, axis=1)  # shape: (N, T, C)
    epistemic = np.einsum('ntc,ntd->ncd', temp, temp) / T  # (N, C, C)

    # We take only the diagonal (variance per class) for simplicity
    epistemic_diag = np.array([np.diag(epistemic[i]) for i in range(N)])  # shape: (N, C)

    # Aleatoric uncertainty (data noise):
    # Equation (14), first term:
    # (1/T) ∑_t diag(p̂_t) - p̂_t p̂_t^T
    # Approximation used: diag(p̄) - (1/T) ∑_t (p̂_t p̂_t^T)
    aleatoric = np.einsum('ntc,ntd->ncd', probs_np, probs_np) / T  # shape: (N, C, C)

    # diag(p̄) - (1/T) ∑_t p̂_t p̂_t^T → keep only the diagonal
    aleatoric = np.array([
        np.diag(np.diag(np.diag(p_bar[i]) - aleatoric[i])) for i in range(N)
    ])  # shape: (N, C)

    return p_bar, epistemic_diag, aleatoric

def compute_uncertainties_scalar(logits, use_softplus=False):
    """
    Compute scalar epistemic and aleatoric uncertainties per image by averaging over classes.

    Args:
        logits: numpy array of shape (N, T, C)
        use_softplus: whether to use Softplus normalization (like PyTorch normalized=True)

    Returns:
        pred_mean_scalar: shape (N,) average predictive probability (optional)
        epistemic_scalar: shape (N,) average epistemic uncertainty scalar
        aleatoric_scalar: shape (N,) average aleatoric uncertainty scalar
    """
    p_bar, epistemic_diag, aleatoric = compute_uncertainties(logits, use_softplus)

    # Average over classes (axis=1)
    pred_mean_scalar = np.mean(p_bar, axis=1)
    # this will essentially yield the probability for a class, so 10 classes = 1/10 to 0.1, not sure why but not too important atm

    epistemic_scalar = np.mean(epistemic_diag, axis=1)
    aleatoric_scalar = np.mean(aleatoric, axis=1)

    return pred_mean_scalar, epistemic_scalar, aleatoric_scalar

# Example usage:
if __name__ == "__main__":
    filepath = "gradcam_outputs/mnist_gradcam_outputs2.h5"
    logits = load_logits_from_h5(filepath)  # shape: (500, 100, 10)

    # Compute uncertainty (change use_softplus=True if needed)
    preds, epistemic_u, aleatoric_u = compute_uncertainties(logits, use_softplus=False)

    print("Predicted class probs:", preds.shape)  # (500, 10)
    print("Epistemic uncertainty:", epistemic_u.shape)  # (500, 10)
    print("Aleatoric uncertainty:", aleatoric_u.shape)  # (500, 10)

    pred_avg, epi_avg, ale_avg = compute_uncertainties_scalar(logits, use_softplus=False)

    # Average over all images to get a dataset-level scalar
    dataset_epistemic_uncertainty = np.mean(epi_avg)
    dataset_aleatoric_uncertainty = np.mean(ale_avg)
    # dataset_pred_prob = np.mean(pred_avg)  # optional

    print("Dataset-level epistemic uncertainty:", dataset_epistemic_uncertainty)
    print("Dataset-level aleatoric uncertainty:", dataset_aleatoric_uncertainty)
    # print("Dataset average predicted probability:", dataset_pred_prob)
