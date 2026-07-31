# 🏗️ Concrete Crack Detector

A Streamlit web app that classifies images of concrete surfaces as **Cracked** or **Non-cracked**, using two trained image classification models (a custom CNN and a MobileNetV3 transfer-learning model). Built to help make manual visual inspection of structures like bridges and pavements faster and more consistent.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#️-tech-stack)
- [Dataset](#-dataset)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Repository Structure](#-repository-structure)
- [Model Performance](#-model-performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview
Concrete cracking is an early warning sign of structural degradation in bridges, buildings, and pavements. Manual visual inspection is slow, subjective, and can be unsafe. This project trains CNN-based image classifiers on the public **Concrete Crack Images for Classification** dataset and serves them through a simple Streamlit interface, so a user can upload a photo and get an instant Cracked / Non-cracked prediction from either model.

---

## ✨ Key Features
- **Binary image classification** — Cracked vs. Non-cracked, via a Streamlit upload UI.
- **Two models to compare** — a custom-trained CNN (`custom_cnn.h5`) and a MobileNetV3 transfer-learning model (`mobilenetv3_transfer.h5`), loaded side by side so you can compare predictions.
- **Notebook-driven training pipeline** — data loading, preprocessing, training, and evaluation are documented in the included Jupyter notebooks.

> Note: bounding-box localization, video/batch processing, and a REST API are not implemented yet — see [Roadmap](#-roadmap) if you're interested in adding them.

---

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **Deep Learning:** TensorFlow / Keras
- **Web Interface:** Streamlit
- **Data & Evaluation:** NumPy, Pandas, scikit-learn (metrics, train/test split)
- **Visualization:** Matplotlib, Seaborn
- **Image Handling:** Pillow (PIL)

*(Reflects what's actually imported in `requirements.txt` and the training notebooks — update this list if the stack changes.)*

---

## 📊 Dataset
Trained and validated on the **Concrete Crack Images for Classification** dataset — 40,000 images (227×227 px), split evenly into `Positive` (cracked) and `Negative` (uncracked) classes.
Source: [Concrete Crack Images for Classification, Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (Özgenel & Sorguç).

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- ~1 GB free disk space (repo includes two pretrained `.h5` models)

### 1. Clone the repository
```bash
git clone https://github.com/Anastasia-2002/Concrete-Crack-Detector.git
cd Concrete-Crack-Detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
Streamlit will open the app in your browser (default: `http://localhost:8501`).

---

## 📖 Usage Guide
1. Launch the app with `streamlit run app.py`.
2. Upload a concrete surface image (JPG/PNG) using the file uploader in the sidebar.
3. The app loads the custom CNN and MobileNetV3 models and runs both on your image.
4. Each model's prediction (**Cracked** / **Non-cracked**) is displayed, so you can compare results.

---

## 📁 Repository Structure
```
Concrete-Crack-Detector/
├── models/
│   ├── custom_cnn.h5              # Custom CNN model weights
│   └── mobilenetv3_transfer.h5    # MobileNetV3 transfer-learning model weights
├── Concrete_Crack_Detection.ipynb # Main training/EDA notebook
├── 23_EG_CE_018_L10.ipynb         # Additional experiment notebook
├── app.py                         # Streamlit application entry point
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## 📈 Model Performance
_Not yet published in this README._ Accuracy/precision/recall/F1 for both models are computed in `Concrete_Crack_Detection.ipynb` (via `sklearn.metrics`) — pull the final values from there and drop them in a table here, e.g.:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Custom CNN | TBD | TBD | TBD | TBD |
| MobileNetV3 (transfer) | TBD | TBD | TBD | TBD |

---

## 🗺️ Roadmap
- [ ] Add a `LICENSE` file
- [ ] Publish real evaluation metrics for both models
- [ ] Bounding-box / defect localization output
- [ ] Batch and video-frame processing
- [ ] REST API endpoint for programmatic access
- [ ] Move model loading/inference logic out of `app.py` into a `src/` module for testability

---

## 🤝 Contributing
Issues and PRs are welcome. If you're adding a feature, please open an issue first to discuss scope, and keep the README's Tech Stack / Repository Structure sections in sync with any changes.

---

## 📄 License
No license file is currently included in this repository, so default copyright applies (all rights reserved) until one is added. If you intend this project to be open source, consider adding an [MIT](https://choosealicense.com/licenses/mit/) or similar license.
