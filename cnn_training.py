import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

#parse XML files and extract coordinates
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    coordinates = []
    for obj in root.findall('.//object'):
        for pt in obj.findall('.//part'):
            x = int(pt.find('x').text)
            y = int(pt.find('y').text)
            coordinates.append((x, y))

    return coordinates

# Function to load and preprocess images and annotations
def load_data(data_dir):
    images = []
    coordinates = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(data_dir, filename)
            xml_path = os.path.join(data_dir, os.path.splitext(filename)[0] + '.xml')

            # Load image
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))  # Resize images as needed
            images.append(img)

            # Parse XML for coordinates
            coords = parse_xml(xml_path)
            coordinates.append(coords)

    images = np.array(images) / 255.0  # Normalize pixel values to be between 0 and 1
    coordinates = np.array(coordinates)

    return images, coordinates

# data
data_dir = 'C:\\Users\\Arvin Asgari\\Desktop\\d'

# Load and split the dataset
images, coordinates = load_data(data_dir)
train_images, test_images, train_coordinates, test_coordinates = train_test_split(images, coordinates, test_size=0.1, random_state=42)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# 
model.add(layers.Dense(0))  # take the number of cordinates

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with a batch size of 1
model.fit(train_images, train_coordinates, epochs=10, validation_data=(test_images, test_coordinates), batch_size=1)

# Save h5
model.save('facial_feature_model.h5')
