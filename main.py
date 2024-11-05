# import os
# import cv2
# import numpy as np
# import yaml
# import tempfile
# from ultralytics import YOLO

# # Filbaner for trenings-, validerings- og testbilder
# train_images_path = "./train/images"
# test_images_path = "./test/images"
# val_images_path = "./val/images"

# # Filbaner for labels
# train_folder_label = "./train/labels"
# test_folder_label = "./test/labels"
# val_folder_label = "./val/labels"

# yaml_file_path = "./data.yaml"

# # Last inn YOLO-modellen med forhåndstrente vekter
# weights_path = "weights/yolov8n.pt"
# model = YOLO(weights_path)

# # Tren YOLO-modellen
# epochs = 2
# learning_rate = 0.0001

# try:
#     print(f"Starter trening med YAML: {yaml_file_path}")
#     results = model.train(data=yaml_file_path, epochs=epochs, lr0=learning_rate, batch=40, imgsz=640)  # Redusert batch size
#     #iou_score = results.box.map75
#     #print(iou_score)
# finally:
#     # os.remove(yaml_file_path)  # Ikke slett YAML-filen
#     pass

import os
import cv2
import random
from ultralytics import YOLO

# Paths for test images
test_images_path = "./test/images"

# Load YOLO model with pretrained weights
weights_path = "weights/yolov8n.pt"
model = YOLO(weights_path)

# Function to display image with bounding boxes
def display_image_with_boxes(image_path, model):
    image = cv2.imread(image_path)

    # Perform object detection
    results = model.predict(source=image)
    predictions = results[0]  # Get predictions for this image

    for box in predictions.boxes:
        # Get bounding box coordinates and confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        
        # Draw bounding box and confidence score on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{box.cls} {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Display six random test images with bounding boxes in separate OpenCV windows
test_images = os.listdir(test_images_path)
random_images = random.sample(test_images, 6)

for i, img_name in enumerate(random_images):
    img_path = os.path.join(test_images_path, img_name)
    img_with_boxes = display_image_with_boxes(img_path, model)
    
    # Show the image in a new OpenCV window
    window_name = f"Image {i+1}: {img_name}"
    cv2.imshow(window_name, img_with_boxes)

# Wait for a key press to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
