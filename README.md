# 🦺 Helmet, Vest, and Head Detection with YOLOv8 🚧

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/) [![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green?logo=github)](https://github.com/ultralytics/ultralytics)

---

This project provides an end-to-end pipeline for detecting **safety helmets, vests, and heads** in images and videos using the powerful YOLOv8 deep learning model. It includes Jupyter notebooks for both training a custom model and running inference on video files.

---

## ✨ Features
- 🎯 Detects three classes: **Helmet, Vest, and Head**
- 🏗️ Custom YOLOv8 training pipeline
- 🎥 Inference on video files with output video generation
- 📒 Easy-to-follow Jupyter notebooks

---

## 📁 Project Structure
```
├── head-helmet-vest-detection-yolov8.ipynb   # Training notebook
├── safty_notebook.ipynb                      # Inference notebook (video)
├── Weights of model.txt                      # Info about model weights
├── README.md                                 # Project documentation
├── .gitattributes, .git/                     # Git config files
```

---

## 🚀 Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yossefmohammed/HELEMT_VEST_HEAD-Detection.git
   cd HELEMT_VEST_HEAD-Detection
   ```
2. **Install dependencies:**
   - Python 3.11 recommended
   - Install required packages:
     ```bash
     pip install ultralytics opencv-python numpy
     ```

---

## 🗂️ Dataset Preparation
- Prepare your dataset with images and labels for the three classes: Helmet, Vest, Head.
- Create a `data.yaml` file specifying paths to your training and validation images, and class names. Example:
  ```yaml
  train: /path/to/train/images
  val: /path/to/val/images
  nc: 3
  names: ['Helmet', 'Vest', 'Head']
  ```

---

## 🏋️‍♂️ Model Training
- Use the [Training Notebook](https://github.com/Yossefmohammed/HELEMT_VEST_HEAD-Detection/blob/main/head-helmet-vest-detection-yolov8.ipynb) to train your model.
- The notebook will:
  - 📦 Install Ultralytics YOLOv8
  - 🧠 Load a pretrained YOLOv8 model (e.g., `yolov8l.pt`)
  - 🏃‍♂️ Train on your dataset (default: 15 epochs, image size 640, batch size 16)
  - 💾 Save results and model weights

---

## 🎬 Inference (Video Detection)
- Use the [Inference Notebook](https://github.com/Yossefmohammed/HELEMT_VEST_HEAD-Detection/blob/main/safty_notebook.ipynb) to run detection on video files.
- The notebook will:
  - 🧠 Load your trained YOLOv8 model (update the path as needed)
  - 🎞️ Process a video file frame by frame
  - 💡 Output a video with detection results drawn on each frame
- Example paths to update in the notebook:
  ```python
  model = YOLO(r"path/to/your_trained_model.pt")
  cap = cv2.VideoCapture(r"path/to/your_video.mp4")
  ```
- Output video will be saved to an `output/` directory.

---

## 📊 Results
- Example training results (from notebook):
  - **mAP50:** ~0.899
  - **mAP50-95:** ~0.569
- Visualizations of training and detection results are available in the notebooks.

---

## 🙏 Acknowledgements
- Built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Dataset and pretrained weights as referenced in the notebooks

---

## 🔗 Social & Project Links
- **LinkedIn Project Post:** [Helmet, Vest and Head Detection with YOLOv8 – LinkedIn](https://www.linkedin.com/posts/yossef-mohammed-358802275_yolov8-computervision-objectdetection-activity-7333495467206397952-mSVU?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEM3f3QBVoyQWy2VA_nA3nv9_e7gGJ3Answ)
- **GitHub Repository:** [HELEMT_VEST_HEAD-Detection](https://github.com/Yossefmohammed/HELEMT_VEST_HEAD-Detection)
- **Email:** ypssefmohammedahmed
- **Phone:** 01126078938

Feel free to open issues or contribute to improve this project!

---
