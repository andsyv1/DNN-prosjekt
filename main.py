import os
import cv2
import numpy as np
import yaml
import tempfile
from ultralytics import YOLO
from convert import convert_annotations  # Importer konverteringsfunksjonen
import inferrence 

# Filbaner for trenings-, validerings- og testbilder
train_images_path = "/home/andreases/DNN/AS_2_4/data/train/images"
val_images_path = "/home/andreases/DNN/AS_2_4/data/val/images"
test_images_path = "/home/andreases/DNN/AS_2_4/data/test/images"

# Filbaner for JSON-filer (annotasjoner)
train_json_path = "/home/andreases/DNN/AS_2_4/train_json/train_label.txt"
val_json_path = "/home/andreases/DNN/AS_2_4/val_json/val_label.txt"

# Filbaner for YOLO-labels (etter konvertering)
train_labels_output_folder = "/home/andreases/DNN/AS_2_4/data/train/labels"
val_labels_output_folder = "/home/andreases/DNN/AS_2_4/data/val/labels"

# Steg 1: Konverter JSON-annotasjoner til YOLO-format (bounding box og segmentering)
convert_annotations(train_json_path, train_labels_output_folder, train_images_path)
convert_annotations(val_json_path, val_labels_output_folder, val_images_path)

# Definer dataset for YOLO med segmentering
dataset = {
    'train': train_images_path,
    'val': val_images_path,
    'nc': 4,  # Antall klasser (ballonger, mennesker, etc.)
    'names': ['balloon', 'not balloon', 'human', 'not human']  # Klassenavn
}

# Generer en YAML-fil for treningen
def generate_yaml_file(dataset):
    yaml_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    with open(yaml_file.name, 'w') as f:
        yaml.dump({
            'train': dataset['train'],  # Bane til treningsbilder
            'val': dataset['val'],      # Bane til valideringsbilder
            'nc': dataset['nc'],        # Antall klasser
            'names': dataset['names']   # Klassenavn
        }, f)
    return yaml_file.name

# Last inn YOLO-modellen med forhåndstrente vekter for segmentering
weights_path = "yolov8s-det.pt"  # Pretrained weights for segmentation
model = YOLO(weights_path)

# Tren YOLO-modellen
epochs = 20
learning_rate = 0.0001
yaml_file_path = generate_yaml_file(dataset)

try:
    print(f"Starter trening med YAML: {yaml_file_path}")
    model.train(data=yaml_file_path, epochs=epochs, lr0=learning_rate)
finally:
    os.remove(yaml_file_path)  # Slett YAML-filen etter trening
