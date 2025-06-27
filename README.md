This project implements a Convolutional Neural Network (CNN) based on the LeNet-5 architecture to detect whether a person is wearing a face mask or not using grayscale images. It uses TensorFlow/Keras and is trained on a custom dataset structured into train, validation, and test folders.

All images are grayscale and resized to 64x64 pixels for model input.

## 🧠 Model Architecture

The model is based on LeNet-5, adapted for binary classification:

    Conv2D → ReLU → MaxPooling

    Conv2D → ReLU → MaxPooling

    Flatten

    Fully Connected Layer (120) → ReLU

    Fully Connected Layer (84) → ReLU

    Output Layer (1 neuron, Sigmoid activation)

    Loss Function: binary_crossentropy
    Optimizer: Adam
    Metrics: accuracy

## 🔁 Data Augmentation

Data augmentation is applied to the training data for improved generalization:

    Rescaling (1./255)

    Rotation

    Width/Height shifts

    Shear

    Zoom

    Horizontal flip

## ✅ Results

The model achieves accurate classification of images into:

    With Mask

    Without Mask

    📌 Final test accuracy depends on dataset quality, size, and augmentation.

## 📦 Dependencies

    tensorflow

    numpy


## 📸 Sample Applications

This system can be integrated into:

    Surveillance systems

    Public entrance monitors

    Workplace safety dashboards

