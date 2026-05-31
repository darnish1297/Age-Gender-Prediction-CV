
# 🧑‍💻 Real-Time Age & Gender Prediction Web App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-ff7c00.svg)

An end-to-end Machine Learning pipeline and web application that predicts a user's age and gender in real-time from webcam photos. The custom Convolutional Neural Network (CNN) was trained from scratch and deployed using a fault-tolerant Gradio web interface.

## 🧠 Engineering Highlights & Architecture
This is not a standard tutorial clone; the model and pipeline feature several production-level optimizations:

* **Custom CNN Architecture (V2):** Designed and trained a deep CNN from scratch with dual-output heads (Regression for Age, Classification for Gender).
* **Advanced Loss Functions:** Upgraded the training loop to use `BCEWithLogitsLoss` for numerical stability in gender classification, and `SmoothL1Loss` (Huber Loss) to prevent gradient bouncing during age regression.
* **Computer Vision Preprocessing:** Utilized OpenCV's Haar Cascades for rapid face detection, engineered with a custom **40% dynamic padding margin** to ensure the network evaluates the entire head structure (jawline, hair, shadows) rather than just tight facial features.
* **Heuristic Calibration for Dataset Bias:** Diagnosed heavy youth-bias in the original UTKFace training dataset and implemented a post-processing heuristic calibration offset to accurately predict real-world adult ages.
* **Confidence Score Engineering:** Addressed "boundary jitter" by manually applying a Sigmoid activation function post-prediction to extract and display the model's exact confidence percentage, providing transparency to the end user.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV (`cv2`), NumPy
* **Web Deployment:** Gradio
* **Dataset:** UTKFace (20,000+ images)

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/darnish1297/Age-Gender-Prediction-CV.git](https://github.com/darnish1297/Age-Gender-Prediction-CV.git)
   cd Age-Gender-Prediction-CV

```

2. **Install the required dependencies:**
*(Ensure you have PyTorch installed for your specific hardware first)*
```bash
pip install gradio opencv-python numpy torch

```


3. **Launch the Web App:**
```bash
python app.py

```


4. **Test it out:** Open the local URL provided in your terminal, allow camera access, ensure your face is well-lit, and click Submit!

## 📈 Future Improvements

* **Dataset Balancing:** The UTKFace dataset has a long-tail distribution heavily skewed toward ages 0-4. Future iterations will involve retraining on a more balanced adult dataset (like IMDB-WIKI) or applying a weighted loss function during training to permanently eliminate the need for heuristic age calibration.
* **Live Video Streaming:** Upgrading the Gradio interface to process continuous frame-by-frame video streams in real-time edge environments.

```

```
