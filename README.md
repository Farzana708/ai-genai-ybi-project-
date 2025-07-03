# ai-genai-ybi-project-
AI internship project - Image classification using Deep Learning
Image Recognition and Classification using Deep Learning

A powerful Convolutional Neural Network (CNN) based project to automatically recognize and classify images using Python and TensorFlow/Keras.

Overview

This project was developed as part of the Deep Learning Internship at YBI Foundation. It demonstrates how deep learning can be leveraged to build an image classification model capable of recognizing images with high accuracy using CNNs.

The model is trained on a standard dataset (like CIFAR-10) and can be extended to custom datasets for real-world use.

Objectives

Understand the fundamentals of CNN and image processing

Preprocess and prepare image data for deep learning models

Design and train a CNN model using TensorFlow/Keras

Evaluate model performance and analyze results

Build a robust image classification pipeline

Tech Stack & Tools

Language : Python
Framework : TensorFlow / Keras
Libraries : NumPy, Matplotlib, Seaborn
Platform : Google Colab / Jupyter Notebook
Dataset : CIFAR-10 (or custom dataset)

Project Structure

model/
├── image_classifier_cnn.h5 → Trained CNN model

dataset/
├── images/ → Sample or custom image data

notebooks/
├── Image_Classification.ipynb → Jupyter notebook with full code

screenshots/
├── training_accuracy.png

README.md
requirements.txt

Methodology

Data Preprocessing: Resize, normalize, and augment image data

Model Architecture: CNN with Conv2D, MaxPooling, Dense, Dropout layers

Training: Use Adam optimizer and categorical_crossentropy loss

Evaluation: Accuracy, loss curves, confusion matrix

Prediction: Model inference on new/unseen images

Sample Code

model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
MaxPooling2D(2,2),
Conv2D(64, (3,3), activation='relu'),
MaxPooling2D(2,2),
Flatten(),
Dense(64, activation='relu'),
Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

Results

Achieved ~88% accuracy on the CIFAR-10 dataset

Low validation loss after training

Confusion matrix analysis reveals strong performance across most classes

Real-time image predictions verified on new samples

Challenges Overcome

Managed memory issues using Google Colab GPUs

Addressed overfitting using Dropout layers and Data Augmentation

Tuned hyperparameters for optimal performance

Future Enhancements

Integrate Transfer Learning (ResNet, VGG16, MobileNet)

Build a web app interface for live image prediction

Expand to multi-label or object detection use cases

References

https://keras.io

https://www.tensorflow.org/tutorials

LeCun, Y. et al. (1998), “Gradient-Based Learning Applied to Document Recognition”

Krizhevsky, A. et al. (2012), “ImageNet Classification with Deep Convolutional Neural Networks”


