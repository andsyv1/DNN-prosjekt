import json
import os

# Funksjon for å konvertere polygon til YOLO bounding box
def convert_polygon_to_yolo_format(all_points_x, all_points_y, img_width, img_height):
    xmin, xmax = min(all_points_x), max(all_points_x)
    ymin, ymax = min(all_points_y), max(all_points_y)

    # Beregn senter, bredde og høyde i forhold til bilde
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return x_center, y_center, width, height

# Funksjon for å konvertere polygon til YOLO-format segmentering (polygon data)
def convert_polygon_to_yolo_segment(all_points_x, all_points_y, img_width, img_height):
    segmentation = []
    for x, y in zip(all_points_x, all_points_y):
        segmentation.append(x / img_width)
        segmentation.append(y / img_height)
    return segmentation

# Funksjon for å laste JSON-filen og konvertere den til YOLO .txt-format med segmentering
def convert_annotations(json_file_path, output_folder, img_folder):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Opprett output folder hvis den ikke eksisterer
    os.makedirs(output_folder, exist_ok=True)

    for img_id, img_data in data.items():
        img_filename = img_data['filename']
        img_path = os.path.join(img_folder, img_filename)
        
        # Finn bildestørrelsen (her bruker vi OpenCV)
        if os.path.exists(img_path):
            import cv2
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
            
            txt_filename = img_filename.replace('.jpg', '.txt')
            txt_file_path = os.path.join(output_folder, txt_filename)
            
            with open(txt_file_path, 'w') as label_file:
                for region in img_data['regions'].values():
                    all_points_x = region['shape_attributes']['all_points_x']
                    all_points_y = region['shape_attributes']['all_points_y']

                    # Konverter polygon til YOLO-format bounding box
                    x_center, y_center, width, height = convert_polygon_to_yolo_format(all_points_x, all_points_y, img_width, img_height)

                    # Konverter polygon til YOLO-format segmentering
                    segmentation = convert_polygon_to_yolo_segment(all_points_x, all_points_y, img_width, img_height)

                    # YOLO-format (klasse ID, senter X, senter Y, bredde, høyde, segmentering)
                    label_file.write(f"0 {x_center} {y_center} {width} {height} " + " ".join(map(str, segmentation)) + "\n")

            print(f"Converted {txt_filename} to YOLO format with segmentation.")
        else:
            print(f"Bildet {img_filename} finnes ikke i {img_folder}.")
