 Brain Tumor Detection System

An AI-powered web application that classifies brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, or No Tumor**. This project utilizes Transfer Learning with a fine-tuned VGG16 architecture to assist in medical image analysis.

 Live Demo
https://brain-tumor-classification-using-deeplearning.streamlit.app/

Key Features
* Real-time Classification: Upload an MRI scan and get instant results.
* VGG16 Architecture: Leverages pre-trained ImageNet weights for superior feature extraction.
* Deep Learning Pipeline: Includes custom data augmentation (brightness adjustment) and intensity normalization.
* High Accuracy: Achieved a test accuracy of ~85%.

 Methodology
1. Data Preprocessing: Images were resized to 128x128 and normalized (pixel values scaled to 0-1).
2. Transfer Learning: Used the VGG16 base model with frozen early layers and unfrozen final convolutional blocks for fine-tuning.
3. Training: Implemented a custom data generator to handle memory-efficient batch processing.

 Tech Stack
* Language: Python
* Deep Learning: TensorFlow / Keras
* Web Framework: Streamlit
* Image Processing: Pillow (PIL), NumPy
* Version Control: Git LFS (Large File Storage for the model)

 Installation & Local Setup
To run this project locally, follow these steps:
Clone the repository
pip install -r requirements.txt
streamlit run main.py