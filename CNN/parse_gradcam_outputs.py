import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
from scipy.special import softmax
from scipy.stats import spearmanr, entropy


def inspect_h5_file(path):
    with h5py.File(path, "r") as f:
        print("Available datasets:")
        for key in f.keys():
            print(f"  - {key}: {f[key].shape}")

        test_x = f["test_x"][:]
        test_y = f["test_y"][:]
        heatmaps = f["heatmaps"][:]  # (num_images, times_to_sample, num_classes, H, W)
        logits = f["logits"][:]  # (num_images, times_to_sample, num_classes)

    return test_x, test_y, heatmaps, logits


def visualize_sample(test_x, test_y, heatmaps, logits, sample_index=0, pass_index=0):
    image = test_x[sample_index]
    label = test_y[sample_index]
    cam = heatmaps[sample_index, pass_index]  # (num_classes, H, W)
    pred = logits[sample_index, pass_index]

    print(f"Grad-CAM heatmap shape: {cam.shape}")
    print(f"True label (one-hot): {label}")
    print(f"Predicted logits: {pred}")
    predicted_class = np.argmax(pred)
    print(f"Predicted class: {predicted_class}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam[predicted_class], cmap="hot")
    plt.title(f"Grad-CAM (class {predicted_class})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(pred)
    plt.title("Logits")
    plt.xlabel("Class")
    plt.ylabel("Logit Value")

    plt.tight_layout()
    plt.show()


def visualize_all_class_heatmaps_for_sample(test_x, test_y, heatmaps, logits, sample_index=0):
    image = test_x[sample_index]
    label = test_y[sample_index]

    num_passes = heatmaps.shape[1]
    num_classes = heatmaps.shape[2]
    H, W = heatmaps.shape[3], heatmaps.shape[4]

    avg_heatmaps = np.mean(heatmaps[sample_index], axis=0)  # (num_classes, H, W)
    var_heatmaps = np.var(heatmaps[sample_index], axis=0)   # (num_classes, H, W)

    sample_logits = logits[sample_index]
    probs = softmax(sample_logits, axis=-1)

    input_H, input_W = image.shape[:2]
    fig = plt.figure(figsize=(2.5 * (num_classes + 1), 14))
    gs = fig.add_gridspec(6, num_classes + 1, height_ratios=[0.3, 2, 0.3, 2, 1.2, 1.2], hspace=0.3)

    # Row title: Avg Grad-CAM
    ax_row1_title = fig.add_subplot(gs[0, 1:])
    ax_row1_title.axis("off")
    ax_row1_title.text(0.5, 0.5, "Average Grad-CAM Overlays", fontsize=14, ha='center', va='center')

    # Row title: Variance
    ax_row2_title = fig.add_subplot(gs[2, 1:])
    ax_row2_title.axis("off")
    ax_row2_title.text(0.5, 0.5, "Variance of Grad-CAM Overlays", fontsize=14, ha='center', va='center')

    # Input image
    ax_input = fig.add_subplot(gs[1, 0])
    ax_input.imshow(image.squeeze(), cmap="gray")
    ax_input.set_title("Input")
    ax_input.axis("off")

    for class_idx in range(num_classes):
        zoom_factor_h = input_H / H
        zoom_factor_w = input_W / W
        avg_resized = zoom(avg_heatmaps[class_idx], (zoom_factor_h, zoom_factor_w), order=1)
        var_resized = zoom(var_heatmaps[class_idx], (zoom_factor_h, zoom_factor_w), order=1)

        avg_resized = (avg_resized - avg_resized.min()) / (avg_resized.max() - avg_resized.min() + 1e-8)
        var_resized = (var_resized - var_resized.min()) / (var_resized.max() - var_resized.min() + 1e-8)

        # Avg Grad-CAM
        ax_avg = fig.add_subplot(gs[1, class_idx + 1])
        ax_avg.imshow(image.squeeze(), cmap="gray")
        ax_avg.imshow(avg_resized, cmap="plasma", alpha=0.6)
        ax_avg.set_title(f"Class {class_idx}")
        ax_avg.axis("off")

        # Variance
        ax_var = fig.add_subplot(gs[3, class_idx + 1])
        ax_var.imshow(image.squeeze(), cmap="gray")
        ax_var.imshow(var_resized, cmap="plasma", alpha=0.6)
        ax_var.set_title(f"Var {class_idx}", fontsize=9)
        ax_var.axis("off")

    # Probabilities
    ax_prob = fig.add_subplot(gs[4, 1:])
    ax_prob.boxplot(probs, vert=True)
    ax_prob.set_title("Softmax Probabilities")
    ax_prob.set_xlabel("Class")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xticks(range(1, num_classes + 1))
    ax_prob.set_xticklabels([str(i) for i in range(num_classes)])

    # Logits
    ax_logits = fig.add_subplot(gs[5, 1:])
    ax_logits.boxplot(sample_logits, vert=True)
    ax_logits.set_title("Raw Logits")
    ax_logits.set_xlabel("Class")
    ax_logits.set_ylabel("Logit Value")
    ax_logits.set_xticks(range(1, num_classes + 1))
    ax_logits.set_xticklabels([str(i) for i in range(num_classes)])

    plt.tight_layout()
    plt.show()


def compute_PSNR_Matrix(test_x, heatmaps, index):
    input_image = test_x[index]
    heatmap = heatmaps[index]

    # Average over sampling axis
    avg_heatmaps = np.mean(heatmap, axis=0)  # (num_classes, H, W)
    num_classes = avg_heatmaps.shape[0]

    # Compute PSNR matrix
    psnr_matrix = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                psnr_matrix[i][j] = 0  # or float('inf') if desired
            else:
                data_range = max(avg_heatmaps[i].max(), avg_heatmaps[j].max()) - min(avg_heatmaps[i].min(),
                                                                                     avg_heatmaps[j].min())
                psnr_val = psnr(avg_heatmaps[i], avg_heatmaps[j], data_range=data_range)
                psnr_matrix[i][j] = psnr_val

    # Get PSNR min/max for color scaling
    psnr_array = np.array(psnr_matrix)
    psnr_min = psnr_array.min()
    psnr_max = psnr_array.max()

    # Plotting
    fig, axs = plt.subplots(num_classes + 1, num_classes + 1, figsize=(12, 12))
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            ax = axs[i, j]
            ax.axis('off')

            if i == 0 and j == 0:
                # Top-left: input image
                ax.imshow(input_image, cmap='gray' if input_image.ndim == 2 else None)
                ax.set_title("Input", fontsize=8)
            elif i == 0:
                # Top row: class heatmaps
                ax.imshow(avg_heatmaps[j - 1], cmap='hot')
                ax.set_title(f"Class {j - 1}", fontsize=8)
            elif j == 0:
                # First column: class heatmaps + label
                ax.imshow(avg_heatmaps[i - 1], cmap='hot')
                ax.set_title(f"Class {i - 1}", fontsize=8)  # Title for clarity
                ax.set_ylabel(f"{i - 1}", fontsize=8, rotation=0, labelpad=20, va='center')
            else:
                # PSNR cell: background shade + PSNR text
                psnr_val = psnr_matrix[i - 1][j - 1]
                norm_val = (psnr_val - psnr_min) / (psnr_max - psnr_min + 1e-8)

                ax.imshow(np.full_like(avg_heatmaps[0], norm_val), cmap='viridis', vmin=0, vmax=1)
                ax.text(
                    0.5, 0.5, f"{psnr_val:.2f}",
                    fontsize=8, ha='center', va='center', transform=ax.transAxes,
                    color='white' if norm_val < 0.5 else 'black'
                )

    plt.tight_layout()
    plt.show()


def compare_psnr_to_logits(test_x, test_y, heatmaps, logits, index):
    avg_heatmaps = np.mean(heatmaps[index], axis=0)  # (num_classes, H, W)
    avg_logits = np.mean(logits[index], axis=0)  # (num_classes,)
    true_label = test_y[index]
    if not np.issubdtype(type(true_label), np.integer):
        true_label = int(np.argmax(true_label))

    num_classes = avg_heatmaps.shape[0]

    # Compute PSNR values between true class and all others
    psnr_vals = []
    for i in range(num_classes):
        if i == true_label:
            psnr_vals.append(np.inf)
        else:
            data_range = max(avg_heatmaps[true_label].max(), avg_heatmaps[i].max()) - min(
                avg_heatmaps[true_label].min(), avg_heatmaps[i].min())
            val = psnr(avg_heatmaps[true_label], avg_heatmaps[i], data_range=data_range)
            psnr_vals.append(val)

    # Get rankings (excluding self for PSNR)
    psnr_rank = np.argsort(psnr_vals)[::-1]  # descending
    logit_rank = np.argsort(avg_logits)[::-1]  # descending

    print("\n--- Ranking Comparison ---")
    print(f"True Class: {true_label}")
    print(f"PSNR Ranking (highest to lowest similarity): {psnr_rank}")
    print(f"Logit Ranking (highest to lowest confidence): {logit_rank}")

    # Compute Spearman rank correlation
    # Exclude the true class index from both
    valid_idxs = [i for i in range(num_classes) if i != true_label]
    psnr_subrank = [psnr_vals[i] for i in valid_idxs]
    logit_subrank = [avg_logits[i] for i in valid_idxs]

    rho, pval = spearmanr(psnr_subrank, logit_subrank)
    print(f"Spearman Correlation: {rho:.3f}, p = {pval:.4f}")

    return rho, psnr_rank, logit_rank


"""
The idea here is that pixels with high variance means the values are drastically different. This corresponds to high change
Low variance means the pixel values remained relativly similar

The idea here is if we take the inverse of this, we are able to map pixels that are actually similar to bright spots
and pixels that are different to dark spots.

for example if we have [1,5] and [2, 20], computing variance we get -> [1,15] we see that pixel[0] is similar but pixel[1] is not
so right now we have [1,15], now we take the inverse [1/1 = 1, 1/15], now we plot this and see the pixels that are similar have a higher value so are brighter

The issue here with inversing by 1 is that we should be normalizing based on min and max value

"""


def compute_inverse_variance_matrix(test_x, heatmaps, index):
    input_image = test_x[index]
    heatmap = heatmaps[index]
    avg_heatmaps = np.mean(heatmap, axis=0)  # (num_classes, H, W)
    num_classes, H, W = avg_heatmaps.shape

    input_H, input_W = input_image.shape[-2:] if input_image.ndim == 3 else input_image.shape

    inverse_var_images = [[None for _ in range(num_classes)] for _ in range(num_classes)]

    # Step 1: Compute all inverse variances, collect for normalization
    all_inv_vals = []

    for i in range(num_classes):
        for j in range(num_classes):
            # Diagonal will be identical
            if i == j:
                inv_variance = np.zeros((H, W))  # Max self-similarity
            else:
                stacked = np.stack([avg_heatmaps[i], avg_heatmaps[j]], axis=0)  # (2, H, W)
                variance_map = np.var(stacked, axis=0)  # (H, W)

                inv_variance = 1 / (variance_map + 1e-8)

                # Map the numbers on log scale to scale down larger values into range
                inv_variance = np.log1p(inv_variance)

            inverse_var_images[i][j] = inv_variance
            all_inv_vals.append(inv_variance)


    # Step 2: Plot
    fig, axs = plt.subplots(num_classes + 1, num_classes + 1, figsize=(12, 12))
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            ax = axs[i, j]
            ax.axis('off')

            if i == 0 and j == 0:
                ax.imshow(input_image.squeeze(), cmap='gray')
                ax.set_title("Input", fontsize=8)
            elif i == 0:
                ax.imshow(avg_heatmaps[j - 1], cmap='hot')
                ax.set_title(f"Class {j - 1}", fontsize=8)
            elif j == 0:
                ax.imshow(avg_heatmaps[i - 1], cmap='hot')
                ax.set_title(f"Class {i - 1}", fontsize=8)
                ax.set_ylabel(f"{i - 1}", fontsize=8, rotation=0, labelpad=20, va='center')
            else:
                inv_var_img = inverse_var_images[i - 1][j - 1]
                zoom_factor = input_H / H
                inv_var_img_resized = zoom(inv_var_img, zoom=zoom_factor, order=1)

                ax.imshow(input_image.squeeze(), cmap='gray')
                ax.imshow(inv_var_img_resized, cmap='plasma', alpha=0.7)
                #ax.set_title("1/Var", fontsize=6)

    plt.suptitle("Inverse Variance Consistency Matrix - What Pixels Stayed Changed The Least", fontsize=14)
    plt.tight_layout()
    plt.show()


def compute_variance_matrix(test_x, heatmaps, index):
    input_image = test_x[index]
    heatmap = heatmaps[index]
    avg_heatmaps = np.mean(heatmap, axis=0)  # (num_classes, H, W)
    num_classes, H, W = avg_heatmaps.shape

    input_H, input_W = input_image.shape[-2:] if input_image.ndim == 3 else input_image.shape

    variance_images = [[None for _ in range(num_classes)] for _ in range(num_classes)]

    # Compute pixel-wise variance images
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                variance_images[i][j] = np.zeros((H, W))  # No variance with self
            else:
                stacked = np.stack([avg_heatmaps[i], avg_heatmaps[j]], axis=0)  # (2, H, W)
                variance_map = np.var(stacked, axis=0)  # (H, W)

                # Map numbers to log scale to scale back higher values so everything is visible
                scaled_variance_map = np.log1p(variance_map)

                variance_images[i][j] = scaled_variance_map

    fig, axs = plt.subplots(num_classes + 1, num_classes + 1, figsize=(12, 12))
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            ax = axs[i, j]
            ax.axis('off')

            if i == 0 and j == 0:
                ax.imshow(input_image.squeeze(), cmap='gray')
                ax.set_title("Input", fontsize=8)
            elif i == 0:
                ax.imshow(avg_heatmaps[j - 1], cmap='hot')
                ax.set_title(f"Class {j - 1}", fontsize=8)
            elif j == 0:
                ax.imshow(avg_heatmaps[i - 1], cmap='hot')
                ax.set_title(f"Class {i - 1}", fontsize=8)
                ax.set_ylabel(f"{i - 1}", fontsize=8, rotation=0, labelpad=20, va='center')
            else:
                var_img = variance_images[i - 1][j - 1]
                # norm_img = (var_img - min_var) / (max_var - min_var + 1e-8)

                # Resize the HxW mask to input_H x input_W
                zoom_factor = input_H / H
                var_img_resized = zoom(var_img, zoom=zoom_factor, order=1)
                ax.imshow(input_image.squeeze(), cmap='gray')
                ax.imshow(var_img_resized, cmap='plasma', alpha=0.7)

                ax.set_title(f"Var", fontsize=6)

    plt.suptitle("Variance Between Class Activation Maps - What Pixels Changed The Most", fontsize=14)
    plt.tight_layout()
    plt.show()


# use the psnr from gradcam saliency masks to predict uncertainty
# This is super useful because if it works on determinsitic models, it means there is no reason to create bayesian cnns
# Doesnt really work or not a lot of evidence to support this

# =========================
# Example usage:
# =========================

def plot_avg_gradcam_vs_entropy_by_class(test_x, test_y, heatmaps, logits):
    num_samples = len(test_x)
    avg_cam_values = []
    entropies = []

    for i in range(num_samples):
        avg_logit = np.mean(logits[i], axis=0)  # (num_classes,)
        probs = softmax(avg_logit)
        ent = entropy(probs)

        avg_heatmap = np.mean(heatmaps[i], axis=(0, 1))  # (H, W)
        avg_cam_val = np.mean(avg_heatmap)

        avg_cam_values.append(avg_cam_val)
        entropies.append(ent)

    avg_cam_values = np.array(avg_cam_values)
    entropies = np.array(entropies)
    true_labels = np.argmax(test_y, axis=1)
    pred_labels = np.argmax(logits.mean(axis=1), axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle("Average Grad-CAM vs Entropy for Each Class", fontsize=16)
    palette = sns.color_palette("tab10", 10)

    for cls in range(10):
        ax = axes[cls // 5, cls % 5]
        mask = true_labels == cls
        correct_mask = mask & (pred_labels == true_labels)
        wrong_mask = mask & (pred_labels != true_labels)

        h1 = ax.scatter(entropies[correct_mask], avg_cam_values[correct_mask],
                        color=palette[cls], s=50, alpha=0.7, edgecolor='k', marker='o', label='Correct')
        h2 = ax.scatter(entropies[wrong_mask], avg_cam_values[wrong_mask],
                        color=palette[cls], s=50, alpha=0.7, edgecolor='k', marker='x', label='Wrong')

        ax.set_title(f"Class {cls}")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Avg Grad-CAM")
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Legend in bottom-right corner of each subplot
        ax.legend(loc='lower right', fontsize='small', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




if __name__ == "__main__":
    dataset_name = "mnist"  # change as needed
    file_path = f"gradcam_outputs/{dataset_name}_gradcam_outputs.h5"

    test_x, test_y, heatmaps, logits = inspect_h5_file(file_path)

    samples_min = 0
    samples_max = 50

    # This doesnt really show any correlation between predictions and the gradcam maps
    plot_avg_gradcam_vs_entropy_by_class(test_x, test_y, heatmaps, logits)

    for sample_idx in range(samples_min, samples_max):
        # visualize_sample(test_x, test_y, heatmaps, logits, sample_idx, pass_index=0)
        visualize_all_class_heatmaps_for_sample(test_x, test_y, heatmaps, logits, sample_idx)
        # compute_PSNR_Matrix(test_x, heatmaps, sample_idx)

        # compare_psnr_to_logits(test_x, test_y, heatmaps, logits, sample_idx)
        # The idea here is to see what stays the same
        # compute the variance between two averaged class samples
        # then inverse the variance so pixels that stay close together are assigned a brighter colour
        compute_inverse_variance_matrix(test_x, heatmaps, sample_idx)
        compute_variance_matrix(test_x, heatmaps, sample_idx)
