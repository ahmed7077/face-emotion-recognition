Facial Expression Detection using EfficientNetB0 (FER2013)

This project implements a deep-learning facial expression classification model trained on the FER2013 dataset.
It uses TensorFlow, EfficientNetB0, and a fully memory-safe pixel parser to process the dataset.
The model predicts one of seven emotions from static face images: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Features

Clean TensorFlow-only pipeline without py_function.

EfficientNetB0 backbone with multi-stage training (frozen and fine-tuning).

Fully compatible with VS Code and local execution.

Supports inference on any image.

Optional Haarcascade-based face detection during inference.

Folder Structure
project/
│── src/
│   ├── train.py
│   ├── inference.py
│── models/
│   └── efficientnetb0_fer.h5
│── data/
│   └── fer2013.csv
│── images/
│   └── test_image.jpg
│── README.md
│── requirements.txt

Requirements

Install all dependencies:

pip install -r requirements.txt


Sample requirements.txt:

tensorflow
numpy
opencv-python
pandas
matplotlib

Dataset

Download the FER2013 CSV file and place it under:

data/fer2013.csv


This file contains pixel strings for each 48x48 emotion-labeled face.

Training

Run the following command from the project root:

python src/train.py


Outputs:

Trained EfficientNetB0 model saved as models/efficientnetb0_fer.h5

Printed accuracy and evaluation results

Automatically handles preprocessing, resizing, and batching

Inference on a Single Image

Place an image under:

images/my_image.jpg


Run:

python src/inference.py --image images/my_image.jpg


Outputs:

Predicted emotion label

Probability score

Haarcascade face detection (if face present)

Model Architecture

Backbone: EfficientNetB0 (ImageNet weights)

GlobalAveragePooling

Dense 256 + BatchNorm + Dropout

Softmax output layer (7 classes)

Optimizer: Adam

Loss: SparseCategoricalCrossentropy

Training workflow:

Freeze EfficientNetB0 backbone

Train classification head

Unfreeze last 50 layers of EfficientNetB0

Fine-tune entire model with low learning rate

Troubleshooting
1. TensorFlow errors

Install correct TF version:

pip install tensorflow==2.13

2. OOM (Out of Memory)

Reduce batch size:

BATCH_SIZE = 16

3. No face detected during inference

Your image may already be a cropped face.
The script automatically classifies the whole image if no face is detected.

4. Wrong path errors

Ensure all paths in train.py and inference.py are absolute or correctly relative.

How to Modify for Custom Use

Change model saving path in train.py

Replace FER2013 with your custom CSV by keeping the same format

Adjust dropout or model size
Example: Use EfficientNetB1 or B2 for higher accuracy

License

This project is free for educational and research use.
