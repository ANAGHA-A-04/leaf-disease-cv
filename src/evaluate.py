import os
import torch
import matplotlib.pyplot as plt

from torchvision import models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from dataset import val_loader, CLASS_NAMES

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load ResNet18 architecture
model = models.resnet18(weights=None)

# Replace final layer
model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

# Load best saved weights
model.load_state_dict(
    torch.load("models/resnet18_best.pth", map_location=device)
)

model = model.to(device)
model.eval()

all_preds = []
all_labels = []

# Create folders
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/errors", exist_ok=True)

error_count = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Save misclassified samples
        for i in range(len(images)):

            if preds[i].item() != labels[i].item():

                if error_count < 10:

                    img = images[i].cpu()

                    # Denormalize
                    mean = torch.tensor(
                        [0.485, 0.456, 0.406]
                    ).view(3, 1, 1)

                    std = torch.tensor(
                        [0.229, 0.224, 0.225]
                    ).view(3, 1, 1)

                    img = img * std + mean
                    img = img.permute(1, 2, 0).numpy()

                    plt.figure(figsize=(4, 4))
                    plt.imshow(img)
                    plt.axis("off")

                    true_label = CLASS_NAMES[
                        labels[i].item()
                    ]

                    pred_label = CLASS_NAMES[
                        preds[i].item()
                    ]

                    plt.title(
                        f"True: {true_label}\nPred: {pred_label}"
                    )

                    plt.savefig(
                        f"reports/errors/error_{error_count}.png",
                        bbox_inches="tight"
                    )

                    plt.close()

                    error_count += 1

# Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_preds
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues")

plt.savefig(
    "reports/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# Classification Report
report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES
)

print(report)

with open(
    "reports/classification_report.txt",
    "w"
) as f:
    f.write(report)