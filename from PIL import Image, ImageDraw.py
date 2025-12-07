import cv2
import os
import xml.etree.ElementTree as ET

def create_annotation_xml(filename, coordinates):
    root = ET.Element("annotation")

    for label, (x, y) in coordinates.items():
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "x").text = str(x)
        ET.SubElement(bbox, "y").text = str(y)

    tree = ET.ElementTree(root)
    tree.write(filename)

def label_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # 3 
                eye_left = (x + w // 4 + w // 20, y + h // 2 - h // 10)
                eye_right = (x + 3 * w // 4 - w // 20, y + h // 2 - h // 10)
                nose = (x + w // 2, y + 2 * h // 3)

                # Draw bb
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(image, (eye_left[0] - 5, eye_left[1] - 5), (eye_left[0] + 5, eye_left[1] + 5), (0, 0, 255), 2)
                cv2.rectangle(image, (eye_right[0] - 5, eye_right[1] - 5), (eye_right[0] + 5, eye_right[1] + 5), (0, 0, 255), 2)
                cv2.rectangle(image, (nose[0] - 5, nose[1] - 5), (nose[0] + 5, nose[1] + 5), (0, 0, 255), 2)

                # Save the annotated image
                cv2.imwrite(output_path, image)

                # Create XML annotation file
                xml_annotation_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.xml")
                create_annotation_xml(xml_annotation_path, {"eye_left": eye_left, "eye_right": eye_right, "nose": nose})

if __name__ == "__main__":
    input_folder = "C:\\Users\\Arvin Asgari\\Desktop\\FD"
    output_folder = "C:\\Users\\Arvin Asgari\\Desktop\\d"

    label_images(input_folder, output_folder)
