# 🩺 Diabetic Retinopathy and Macular Edema Detection Using Deep Learning

An AI-powered medical image analysis system that uses Deep Learning to detect **Diabetic Retinopathy (DR)** and classify **Macular Edema (DME) risk** from retinal fundus images.

The application uses a trained multi-output deep learning model and provides predictions through an interactive Streamlit web interface.

## 🚀 Project Overview

Diabetic Retinopathy is a diabetes-related eye disease that can lead to vision impairment if not identified early.

This project aims to assist in retinal image analysis by developing a deep learning-based system that predicts the severity of diabetic retinopathy and the risk level of macular edema.

Users can upload retinal fundus images and receive two prediction results.

## ✨ Key Features

- Upload retinal fundus images in JPG, JPEG and PNG formats.
- Automatic retinal image preprocessing.
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
- ResNet50 preprocessing.
- Deep learning-based image classification.
- Simultaneous prediction of DR and DME risk.
- Display original and preprocessed retinal images.
- Interactive Streamlit interface.

## 🧠 Prediction Categories

### 1. Diabetic Retinopathy Classification

The model predicts five classes:

| Class | Prediction |
|---|---|
| 0 | No Diabetic Retinopathy |
| 1 | Mild Diabetic Retinopathy |
| 2 | Moderate Diabetic Retinopathy |
| 3 | Severe Diabetic Retinopathy |
| 4 | Proliferative Diabetic Retinopathy |

### 2. Macular Edema Classification

The model predicts three categories:

| Class | Prediction |
|---|---|
| 0 | No Macular Edema |
| 1 | Mild Risk of Macular Edema |
| 2 | High Risk of Macular Edema |

## ⚙️ Image Processing Workflow

1. **Image Upload:** The user uploads a retinal fundus image.
2. **Image Resizing:** The image is resized to 256 × 256 pixels.
3. **CLAHE Enhancement:** Improves image contrast using Contrast Limited Adaptive Histogram Equalization.
4. **Image Normalization:** ResNet50 preprocessing is applied to prepare the image for the trained model.
5. **Deep Learning Prediction:** The trained model processes the image and generates two prediction outputs.
6. **Result Visualization:** The application displays the predicted DR severity and DME risk category.

## 🏗️ System Architecture

```text
       Retinal Fundus Image
                |
                v
          Image Upload
                |
                v
        Image Preprocessing
                |
        +-------+-------+
        |               |
    Resizing          CLAHE
    256x256       Enhancement
        |               |
        +-------+-------+
                |
                v
       ResNet50 Preprocessing
                |
                v
       Trained Deep Learning
          Model (.keras)
                |
         +------+------+
         |             |
         v             v
        DR             DME
   Classification   Classification
         |             |
         v             v
    5 Classes       3 Classes
         |             |
         +------+------+
                |
                v
        Streamlit Results
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV |
| Pretrained Architecture Utilities | ResNet50 |
| Numerical Computing | NumPy |
| Image Handling | Pillow |
| Web Framework | Streamlit |
| Model Format | Keras (.keras) |

## 📂 Project Structure

```text
diabetic-retinopathy--DR--and-macular-edema--DME-/
│
├── code_1.py
│
├── model1.keras
│
└── README.md
```

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible TensorFlow-supported version.
- pip package manager.
- VS Code or any Python IDE.

### Step 1: Clone the Repository

```bash
git clone https://github.com/DHANUSHAMBIKAPATHY/diabetic-retinopathy--DR--and-macular-edema--DME-.git
```

### Step 2: Navigate to the Project Directory

```bash
cd diabetic-retinopathy--DR--and-macular-edema--DME-
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install streamlit tensorflow keras numpy opencv-python pillow
```

### Step 5: Run the Application

```bash
streamlit run code_1.py
```

The application will open in your browser.

```text
http://localhost:8501
```

## 📊 Expected Output

The application displays:

- Original uploaded retinal image.
- Preprocessed retinal image.
- Predicted diabetic retinopathy severity.
- Predicted macular edema risk category.

## 🎯 Project Objectives

- Apply deep learning techniques to medical image classification.
- Develop an automated retinal image analysis application.
- Identify different severity levels of diabetic retinopathy.
- Classify macular edema risk levels.
- Integrate a trained deep learning model into a user-friendly web application.

## 🔮 Future Enhancements

- Improve model performance using larger and more diverse retinal image datasets.
- Add prediction confidence scores.
- Develop a patient history management system.
- Integrate explainable AI techniques such as Grad-CAM.
- Deploy the application on a cloud platform.
- Improve the user interface and reporting functionality.

## ⚠️ Medical Disclaimer

This project is developed for educational and research purposes. Its predictions are not a substitute for professional ophthalmological examination, medical diagnosis or treatment. Always consult a qualified eye specialist for medical evaluation.

## 👨‍💻 Developers

**Dhanush A**
GitHub: https://github.com/DHANUSHAMBIKAPATHY

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.
