# 😊 Face Emotion Detection using CNN

A deep learning project to classify facial emotions using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It detects 7 types of facial expressions from grayscale images of size 48x48 pixels.

## 📚 Emotions Detected
- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😄
- Sad 😢
- Surprise 😲
- Neutral 😐

## 🧾 Dataset Structure
Organize your dataset like this:

project/  
├── train/  
│   ├── angry/  
│   ├── disgust/  
│   ├── fear/  
│   ├── happy/  
│   ├── sad/  
│   ├── surprise/  
│   └── neutral/  
├── test/  
│   ├── angry/  
│   ├── disgust/  
│   ├── fear/  
│   ├── happy/  
│   ├── sad/  
│   ├── surprise/  
│   └── neutral/

## 🛠️ Requirements
- Python 3.x  
- TensorFlow  
- Keras  
- NumPy  
- Pandas  
- Matplotlib  

Install with:  
`pip install tensorflow keras numpy pandas matplotlib`

## 🚀 How to Use
1. Put your images in `train/` and `test/` folders.
2. Optionally add pretrained weights as `model.weights.best.keras`.
3. Run the script:  
`python your_script.py`

The script will:
- Count and visualize images
- Show sample emotion images
- Train the CNN model
- Save the best model
- Plot training and validation loss

## 🧠 Model Architecture
Input: 48x48 grayscale images  
[Conv2D + BatchNorm + ReLU + MaxPool] × 3 → Flatten → Dense(128) → Dropout → Dense(7, softmax)  
Optimizer: Adam  
Loss: Categorical Crossentropy  
Metrics: Accuracy  

## 📈 Outputs
- Bar chart of image counts
- Training/Validation loss graph
- Trained model saved as `best_model.h5`

## 🔁 Callbacks
- EarlyStopping(patience=5, restore_best_weights=True)  
- ModelCheckpoint('best_model.h5', save_best_only=True)

## ✅ Notes
- Images must be grayscale and 48x48 size  
- Modify learning rate, batch size, or layers for better results  
- Easily extendable for real-time webcam detection

## 👨‍💻 Author
Made with Fatma mohmmed jamal using Python & TensorFlow  
Feel free to fork, star ⭐, and contribute!
