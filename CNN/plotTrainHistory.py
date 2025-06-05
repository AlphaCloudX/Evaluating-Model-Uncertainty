import os
import json
import matplotlib.pyplot as plt

history_dir = "model_history"
plot_dir = "plot_images"
os.makedirs(plot_dir, exist_ok=True)

history_files = [f for f in os.listdir(history_dir) if f.endswith("_history.json")]
history_files.sort()

# Load all histories
all_histories = {}
for file in history_files:
    dataset_name = file.replace("_history.json", "")
    with open(os.path.join(history_dir, file), "r") as f:
        all_histories[dataset_name] = json.load(f)

# === Combined Accuracy Plot ===
plt.figure(figsize=(12, 6))
for name, history in all_histories.items():
    if "accuracy" in history:
        train_acc = history["accuracy"]
        val_acc = history.get("val_accuracy", [])
    elif "acc" in history:
        train_acc = history["acc"]
        val_acc = history.get("val_acc", [])
    else:
        continue

    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, label=f"{name} Train", linestyle="-")
    plt.plot(epochs, val_acc, label=f"{name} Val", linestyle="--")

plt.title("Training and Validation Accuracy (All Datasets)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "accuracy_plot.png"))
plt.show()

# === Combined Loss Plot ===
plt.figure(figsize=(12, 6))
for name, history in all_histories.items():
    train_loss = history["loss"]
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label=f"{name} Train", linestyle="-")
    plt.plot(epochs, val_loss, label=f"{name} Val", linestyle="--")

plt.title("Training and Validation Loss (All Datasets)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "loss_plot.png"))
plt.show()

# === Individual Plots Per Dataset ===
for name, history in all_histories.items():
    # Accuracy
    plt.figure(figsize=(8, 5))
    if "accuracy" in history:
        train_acc = history["accuracy"]
        val_acc = history.get("val_accuracy", [])
    elif "acc" in history:
        train_acc = history["acc"]
        val_acc = history.get("val_acc", [])
    else:
        continue

    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, label="Train", linestyle="-")
    plt.plot(epochs, val_acc, label="Validation", linestyle="--")
    plt.title(f"{name.upper()} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_accuracy.png"))
    plt.show()
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    train_loss = history["loss"]
    val_loss = history.get("val_loss", [])
    plt.plot(epochs, train_loss, label="Train", linestyle="-")
    plt.plot(epochs, val_loss, label="Validation", linestyle="--")
    plt.title(f"{name.upper()} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_loss.png"))
    plt.show()
    plt.close()
