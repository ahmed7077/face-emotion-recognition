# Facial Expression Detection using EfficientNetB0 (FER2013)

## Overview

This project implements a deep learning-based facial expression classification system trained on the FER2013 dataset. It uses TensorFlow and EfficientNetB0 as the backbone model to classify facial expressions from static images.

The model predicts one of seven emotion classes:
Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The system is designed for efficient training, clean preprocessing, and reliable inference on local systems.

---

## Features

* Deep learning pipeline using TensorFlow and EfficientNetB0
* FER2013 dataset-based emotion classification
* Multi-stage training (feature extraction and fine-tuning)
* Fully local execution compatible with VS Code
* Image-based inference support
* Optional Haarcascade-based face detection during inference
* Memory-safe preprocessing pipeline without py_function dependency

---

## Folder Structure

```text
project/
│── src/
│   ├── train.py
│   ├── inference.py
│
│── models/
│   └── efficientnetb0_fer.h5
│
│── data/
│   └── fer2013.csv
│
│── images/
│   └── test_image.jpg
│
│── README.md
│── requirements.txt
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### requirements.txt

```text
tensorflow
numpy
opencv-python
pandas
matplotlib
```

---

## Dataset

The FER2013 dataset should be placed in:

```text
data/fer2013.csv
```

This dataset contains 48x48 grayscale facial images stored as pixel values with corresponding emotion labels.

---

## Training

Run training using:

```bash
python src/train.py
```

### Output:

* Trained model saved at:

  ```text
  models/efficientnetb0_fer.h5
  ```
* Training accuracy and validation metrics displayed
* Automatic preprocessing and batching

---

## Inference on a Single Image

Place your test image in:

```text
images/my_image.jpg
```

Run inference:

```bash
python src/inference.py --image images/my_image.jpg
```

### Output:

* Predicted emotion label
* Confidence score
* Optional face detection using Haarcascade

---

## Model Architecture

* Backbone: EfficientNetB0 (ImageNet pretrained)
* GlobalAveragePooling layer
* Dense layer (256 units)
* Batch Normalization
* Dropout layer
* Softmax output layer (7 classes)

### Training Strategy:

* Stage 1: Freeze EfficientNetB0 backbone and train classifier head
* Stage 2: Unfreeze last 50 layers for fine-tuning
* Optimizer: Adam
* Loss function: Sparse Categorical Crossentropy

---

## Troubleshooting

### TensorFlow errors

Use compatible version:

```bash
pip install tensorflow==2.13
```

### Out of Memory (OOM)

Reduce batch size:

```python
BATCH_SIZE = 16
```

### No face detected during inference

If the image already contains a cropped face, the model will still perform classification without detection.

### Path issues

Ensure all file paths in `train.py` and `inference.py` are correctly set relative to the project root.

---

## Customization

* Modify model save path in `train.py`
* Replace FER2013 with custom datasets (same format required)
* Upgrade backbone to EfficientNetB1 or B2 for improved accuracy
* Adjust dropout rate for regularization tuning

---

## License

This project is intended for educational and research purposes only.

---

