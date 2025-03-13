# **Skin Lesion Detection App using MobileNetV3**  

![Skin Lesion Detection](https://raw.githubusercontent.com/gopibattineni/skl/main/Skin%20lesion.png)


## **📌 Overview**  
This project is an **AI-powered skin lesion detection application** leveraging **MobileNetV3** for efficient and accurate classification of skin abnormalities. The model is designed to work on **mobile and edge devices**, making real-time detection accessible for dermatological analysis.  

## **🚀 Features**  
✅ Deep learning-based **skin lesion classification**  
✅ **MobileNetV3** for **lightweight & efficient inference**  
✅ Real-time **image processing & prediction**  
✅ **User-friendly UI** for easy image uploads  
✅ **Potential for Telemedicine & Remote Diagnosis**  

## **🛠️ Tech Stack**  
- **Python** (TensorFlow, Keras)  
- **MobileNetV3** (Pre-trained deep learning model)  
- **Flask/FastAPI** (For backend API)  
- **Html/CSS** (For frontend UI – Optional)  
- **OpenCV** (Image preprocessing)  

## **📂 Project Structure**  
```bash
├── dataset/               # Dataset of skin lesion images
├── models/                # Trained MobileNetV3 model
├── app/                   # Backend and frontend code
│   ├── server.py          # API for model inference
│   ├── app.py             # Streamlit-based UI (Optional)
├── notebooks/             # Jupyter notebooks for model training
│   ├── training.ipynb     # Model training and evaluation
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── LICENSE                # License file
```  

## **🖼️ Dataset**  
- The model is trained using **ISIC (International Skin Imaging Collaboration) Dataset**.  
- Preprocessing includes **image resizing, normalization, and augmentation** to improve robustness.  

## **📖 How to Use**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/gopibattineni/skl.git
cd skl
```
### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```
### **3️⃣ Run the App**  
For API-based deployment:  
```bash
python app/server.py
```
For Streamlit-based UI:  
```bash
streamlit run app/app.py
```  

## **🧠 Model Training**  
1. Load the dataset and preprocess images.  
2. Train MobileNetV3 with transfer learning and fine-tuning.  
3. Evaluate on a test set and optimize for mobile deployment.  

## **📊 Results**  
- Achieved **high accuracy and F1-score** in skin lesion classification.  
- Lightweight model allows **real-time inference** on mobile devices.  

## **📌 Future Improvements**  
🔹 Improve model robustness with more diverse datasets.  
🔹 Integrate a **Flutter-based mobile app**.  
🔹 Deploy as a **web service for easy accessibility**.  

## **🤝 Contributing**  
Contributions are welcome! Feel free to fork this repo, submit issues, or open a pull request.  

## **📜 License**  
This project is licensed under the **MIT License**.  
