# Concrete-Crack-Detector
[ Input Concrete Image ]
│
▼
[ Data Preprocessing ] ---> Resizing, Normalization, Noise Reduction
│
▼
[ CNN Feature Extractor ] ---> ResNet50 / Custom Convolutional Layers
│
▼
[ Classification & Masking ] ---> Crack Detection Probability + Region Isolation
│
▼
[ User Interface / API Output ] ---> Visualized Bounding Boxes & Health Report


A web app with a model in built to detect concrete cracks in bridges
Repository Structure
Concrete-Crack-Detector/
├── assets/                  # Sample output images and UI screenshots
├── data/                    # Dataset configuration files
├── models/                  # Saved model checkpoints (.pth / .h5)
├── notebooks/               # Jupyter Notebooks for EDA & Model Experiments
├── src/                     # Core Source Code
│   ├── preprocessing.py     # Image cleaning, transformation, and augmentation
│   ├── model.py             # CNN architecture definition
│   ├── train.py             # Training loop and metric evaluation
│   └── inference.py         # Prediction pipeline for new inputs
├── app.py                   # Main web application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
