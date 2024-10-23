import os
import cv2
import numpy as np
import yaml
import tempfile
from ultralytics import YOLO

# Filbaner for trenings-, validerings- og testbilder
train_images_path = "/home/andreases/DNN/Prosjekt/Data/train/images"
test_images_path = "/home/andreases/DNN/Prosjekt/Data/test/images"
val_images_path = "/home/andreases/DNN/Prosjekt/Data/val/images"

# Filbaner for labels
train_folder_label = "/home/andreases/DNN/Prosjekt/Data/train/labels"
test_folder_label = "/home/andreases/DNN/Prosjekt/Data/test/labels"
val_folder_label = "/home/andreases/DNN/Prosjekt/Data/val/labels"

yaml_file_path = "/home/andreases/DNN/Prosjekt/Data/data.yaml"

# Last inn YOLO-modellen med forhåndstrente vekter
weights_path = "weights/yolov8n.pt"
model = YOLO(weights_path)

# Tren YOLO-modellen
epochs = 20
learning_rate = 0.0001

try:
    print(f"Starter trening med YAML: {yaml_file_path}")
    model.train(data=yaml_file_path, epochs=epochs, lr0=learning_rate, batch=4)  # Redusert batch size
finally:
    # os.remove(yaml_file_path)  # Ikke slett YAML-filen
    pass
