# 🐍 Snake Bite Classification using GPU acceleration

This project is a **deep learning-based classification model** using **ResNet50** and **TensorFlow** to classify **snake bites as Poisonous or Non-Poisonous** based on wound patterns. The model is trained on an image dataset and fine-tuned for better accuracy using GPU.  

---

## 🚀 Features  
✅ **Deep Learning Model**: Uses ResNet50 as a feature extractor.  
✅ **Transfer Learning**: Fine-tuned for improved accuracy.  
✅ **Data Augmentation**: Helps generalization on unseen data.  
✅ **Real-time Predictions**: Classifies images of snake bites.  
✅ **GPU Acceleration**: Supports CUDA for faster training.  

---

## 🛠️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/anis196/snk-bite-det.git
cd snk-bite-det
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Check GPU Availability (Optional)  
```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

### 4️⃣ Prepare the Dataset  
Organize the dataset in the following structure:  
```plaintext
/dataset  
    /Poisonous  
        - image1.jpg  
        - image2.jpg  
    /Non_Poisonous  
        - image1.jpg  
        - image2.jpg  
```
🔹 **Update the dataset path** in `snk.py` before running the script.

---

## 🎯 Model Training & Usage  

### 5️⃣ Train or Load the Model  
If running for the first time, the model will train and save automatically.  
```bash
python snk.py
```
To avoid retraining, the model is saved as `resnet50_snake_bite_classifier.h5` and will be loaded in future runs.

---

## 🐍 Making Predictions  
Use an image file to test the model:  
```python
from snk import predict_image

predict_image("path_to_new_image.jpg", model)
```
### **🔹 Example Output:**  
```plaintext
Predicted: Poisonous 🐍 (Confidence: 0.87)
```
or  
```plaintext
Predicted: Non-Poisonous ✅ (Confidence: 0.93)
```
## 📸 Screenshots of terminal 

![Screenshot 2025-02-11 230048](https://github.com/user-attachments/assets/cc2bba40-cfec-414f-bdeb-bf80363d16a9)

![Screenshot 2025-02-11 230108](https://github.com/user-attachments/assets/1cc1bd32-e7a5-4d17-8cb3-693b9b4f6f09)

![Screenshot 2025-02-11 225851](https://github.com/user-attachments/assets/5635feb0-8543-4032-a2ac-a4bf86247576)


---

## 🛠️ Tools & Technologies Used  
🔹 **Programming:** Python  
🔹 **Frameworks:** TensorFlow, Keras  
🔹 **Libraries:** OpenCV, NumPy, Matplotlib  
🔹 **Database & Storage:** MySQL (if used), Local Storage  
🔹 **Version Control:** Git  

---

## 📜 License  
This project is licensed under the **MIT License**.  

---

## 📬 Contact  
For any queries, reach out at ✉️ [shaikhanis2004@gmail.com](mailto:shaikhanis2004@gmail.com).  

---
