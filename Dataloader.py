# Dataloader.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class FishSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = os.path.abspath(images_dir)  # Absolutt bane for bildemappe
        self.masks_dir = os.path.abspath(masks_dir)    # Absolutt bane for maskemappe
        self.transform = transform
        self.mask_transform = mask_transform

        # Finn alle bildefiler i alle undermapper
        self.image_paths = []
        self.mask_paths = []

        print(f"Bruker bilder-katalog: {self.images_dir}")
        print(f"Bruker masker-katalog: {self.masks_dir}")

        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    
                    # Generer relatert sti og bytt 'fish' til 'mask' for maskestien
                    relative_path = os.path.relpath(image_path, self.images_dir)
                    mask_relative_path = relative_path.replace("fish_", "mask_")  # Endre 'fish_' til 'mask_'
                    mask_path = os.path.join(self.masks_dir, mask_relative_path)

                    # Debug: Skriv ut bilde- og maskestiene for verifikasjon
                    print(f"Bilde: {image_path} -> Forventet maske: {mask_path}")

                    # Legg til bilde og maske hvis masken eksisterer
                    if os.path.exists(mask_path):
                        self.image_paths.append(image_path)
                        self.mask_paths.append(mask_path)
                        print(f"Fant maske: {mask_path}")
                    else:
                        print(f"Advarsel: Fant ikke maske for {image_path}")

        # Debug-utskrift for å vise totalt antall bilder og masker funnet
        print(f"Fant {len(self.image_paths)} bilder og {len(self.mask_paths)} masker i {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask.long()

# Definer transformasjoner for bilder og masker
def get_transforms():
    image_transform = T.Compose([
        T.Resize((60, 60)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = T.Compose([
        T.Resize((60, 60)),
        T.ToTensor()
    ])
    
    return image_transform, mask_transform
