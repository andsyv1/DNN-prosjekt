import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Dataloader import FishSegmentationDataset, get_transforms
from tqdm import tqdm

# Opprett en mappe for å lagre resultater
result_dir = "./result_UNet"
os.makedirs(result_dir, exist_ok=True)

# Få transformasjoner fra Dataloader.py
image_transform, mask_transform = get_transforms()

# Opprett trenings-, validerings- og testdatalastere
train_dataset = FishSegmentationDataset(
    images_dir="./Dataset/train/images",
    masks_dir="./Dataset/train/labels",
    transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = FishSegmentationDataset(
    images_dir="./Dataset/val/images",
    masks_dir="./Dataset/val/labels",
    transform=image_transform,
    mask_transform=mask_transform
)

test_dataset = FishSegmentationDataset(
    images_dir="./Dataset/test/images",
    masks_dir="./Dataset/test/labels",
    transform=image_transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Last ned forhåndstrent U-Net modell og tilpass for binær segmentering (fisk vs bakgrunn)
model = smp.Unet(
    encoder_name="resnet18",        # Encoder-backbone
    encoder_weights="imagenet",     # Forhåndstrente vekter
    classes=1,                      # Én kanal for binær segmentering
    activation=None                 # Ingen aktivering her, vi bruker sigmoid i etterkant
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Definer tapfunksjon og optimizer
criterion = nn.BCEWithLogitsLoss()  # For binær segmentering
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Variabler for å logge trenings- og validerings-tap
train_losses = []
val_losses = []
num_epochs = 10

sigmoid = nn.Sigmoid()  # Sigmoid for å transformere logits til sannsynlighet

# Trenings- og valideringssløyfe
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        masks = masks.float().to("cuda" if torch.cuda.is_available() else "cpu")  # Endre til float for BCEWithLogitsLoss

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks.unsqueeze(1))  # Legg til kanal-dimensjon for BCEWithLogitsLoss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss}")

    # Valideringssløyfe
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            masks = masks.float().to("cuda" if torch.cuda.is_available() else "cpu")

            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            val_loss += loss.item()

            preds = sigmoid(outputs).cpu().numpy()  # Påfør sigmoid for å få sannsynlighet
            preds = (preds > 0.5).astype(np.uint8).flatten()  # Terskel for binær segmentering
            labels = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")

    # Beregn metrikker
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    iou = jaccard_score(all_labels, all_preds, average="binary")

    print(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}, IoU: {iou}")

    # Lagre metrikker til fil
    with open(os.path.join(result_dir, "metrics.txt"), "a") as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}\n")
        f.write(f"Training Loss: {train_loss}\n")
        f.write(f"Validation Loss: {val_loss}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"IoU: {iou}\n\n")

# Funksjon for å visualisere en batch med originale bilder og segmenteringsresultater
def visualize_batch(data_loader, model, dataset_name):
    model.eval()
    images, masks = next(iter(data_loader))
    images = images.to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = sigmoid(model(images)).cpu()  # Påfør sigmoid her

    fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))

    for i in range(len(images)):
        # Originalbilde
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title(f"{dataset_name} - Original Image")
        axes[i, 0].axis("off")

        # Segmenteringsmaske
        mask = outputs[i].squeeze(0).numpy() > 0.5  # Binariser basert på terskel
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(f"{dataset_name} - Segmentation Mask")
        axes[i, 1].axis("off")

        # Overlegg segmenteringsmasken på originalbildet
        original_image = images[i].cpu().permute(1, 2, 0).numpy()
        overlay = np.zeros_like(original_image)
        overlay[..., 1] = mask * 255  # Grønt overlegg
        blended_image = (0.6 * original_image) + (0.4 * overlay / 255)  # Blending for overlegg

        axes[i, 2].imshow(blended_image)
        axes[i, 2].set_title(f"{dataset_name} - Overlayed Segmentation")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{dataset_name.lower()}_segmentation_batch.png"))
    plt.show()

# Visualiser en batch fra hvert datasett
visualize_batch(train_loader, model, "Train")
visualize_batch(val_loader, model, "Validation")
visualize_batch(test_loader, model, "Test")

# Plott trenings- og valideringstap
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig(os.path.join(result_dir, "loss_plot.png"))
plt.show()

# Evaluering på testsettet
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        masks = masks.float().to("cuda" if torch.cuda.is_available() else "cpu")
        
        outputs = sigmoid(model(images)).cpu().numpy()
        preds = (outputs > 0.5).astype(np.uint8).flatten()
        labels = masks.cpu().numpy().flatten()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

# Beregn og lagre sluttmetrikker på testsettet
f1 = f1_score(all_labels, all_preds, average="binary")
precision = precision_score(all_labels, all_preds, average="binary")
recall = recall_score(all_labels, all_preds, average="binary")
iou = jaccard_score(all_labels, all_preds, average="binary")

with open(os.path.join(result_dir, "test_metrics.txt"), "w") as f:
    f.write("Test Set Metrics\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"IoU: {iou}\n")

# Beregn og visualiser forvirringsmatrisen
conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.show()

print("Trening og evaluering ferdig. Resultater lagret i result_UNet-mappen.")
