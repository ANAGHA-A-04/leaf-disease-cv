import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch

from aug_transform import train_transform


def denormalize(tensor):
    """
    Convert normalized tensor back to displayable image.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)

    return tensor.permute(1, 2, 0).numpy()


# Select one sample image
image_path = next(Path("data/train/late_blight").glob("*.jpg"))
# Or specify manually:
# image_path = "data/train/healthy/sample.jpg"

image = Image.open(image_path).convert("RGB")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

# Generate 7 augmented versions
for i, ax in enumerate(axes.flat[1:], start=1):

    aug_img = train_transform(image)

    # Convert tensor back to image
    aug_img = denormalize(aug_img)

    ax.imshow(aug_img)
    ax.set_title(f"Augmented {i}")
    ax.axis("off")

plt.suptitle("Before and After Data Augmentation", fontsize=16)

plt.tight_layout()

plt.savefig(
    "augmentation_grid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: augmentation_grid.png")