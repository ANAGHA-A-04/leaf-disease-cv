import matplotlib.pyplot as plt
from dataset import train_loader

images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    img = images[i].permute(1, 2, 0)
    img = img.numpy()
    img = (img - img.min()) / (img.max() - img.min())
    ax.imshow(img)
    CLASS_NAMES = ["early_blight", "healthy", "late_blight", "leaf_mold"]
    ax.set_title(f"Class: {CLASS_NAMES[labels[i].item()]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_batch.png")
plt.show()