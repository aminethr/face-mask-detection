import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Path to the trained model
MODEL_PATH = "C:/Users/M-Tech/Desktop/github/face_mask/model/face_mask_detection_model.h5"

# Function to preprocess the input image
def preprocess_image(image_path):
    """
    Preprocess the input image for the model.
    - Converts to grayscale
    - Resizes to 64x64
    - Normalizes pixel values to [0, 1]
    """
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to match the model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    return img_array

# Function to predict the class of an image
def predict(image_path):
    # Load the model
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")

    # Preprocess the image
    img_array = preprocess_image(image_path)
    print("Image processed successfully!")

    # Make a prediction
    probabilities = model.predict(img_array)[0][0]  # Extract single prediction
    wearing_mask = (1 - probabilities) * 100
    not_wearing_mask = probabilities * 100
    # Print the results
    print(f"Probability of wearing a mask: {wearing_mask:.2f}%")
    print(f"Probability of not wearing a mask: {not_wearing_mask:.2f}%")

# Main script
if __name__ == "__main__":
    # Path to the image to test
    IMAGE_PATH = "IMAGE PATH"  # Replace with your image path

    # Call the predict function
    predict(IMAGE_PATH)















