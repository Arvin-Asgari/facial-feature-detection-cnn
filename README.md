# Facial Feature Detection System 

**Complete computer vision pipeline**: Automated data annotation → CNN training → real-time webcam detection of eyes/nose positions.

## 📋 Features
- Automated XML annotation of 2000+ face images using OpenCV face detection
- Custom CNN trained with TensorFlow/Keras on annotated dataset
- Real-time eye/nose detection using trained model + webcam feed
- Complete end-to-end pipeline from raw images to live detection

##  Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%2337BC9B?style=flat&logo=TensorFlow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27C1EF?style=flat&logo=opencv&logoColor=white)

##  Quick Start

**1. Install Dependencies**
```bash
pip install tensorflow opencv-python numpy scikit-learn
```

**2. Prepare Dataset**
Place your face images in your dataset folder.

**3. Generate Annotations**
```bash
python data_annotation.py
```

**4. Train CNN Model**
```bash
python cnn_training.py
```

**5. Run Live Detection**
```bash
python live_detection.py
```

## 📁 Project Structure
```text
├── data_annotation.py     # Auto-generates XML labels from images
├── cnn_training.py        # Trains CNN (224x224 input, facial landmarks)
├── live_detection.py      # Real-time webcam detection
└── README.md
```

##  Notes
- **Dataset:** Not included in this repo due to size (2000+ personal images).
- **Model:** Trained on personal face images for specific eye/nose detection.
- **Setup:** Adjust file paths in scripts to match your local directory structure.
