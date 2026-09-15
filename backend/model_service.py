import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# Find the main project folder
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Path to trained model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_model.keras"
)


# Class names must match the training order
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "No-tumor",
    "pituitary"
]


# Load model once when the backend starts
model = load_model(MODEL_PATH)


def predict_image(image: Image.Image):

    # Resize image to the size used during training
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Handle grayscale image
    if len(image_array.shape) == 2:
        image_array = np.stack(
            [image_array] * 3,
            axis=-1
        )

    # Remove alpha channel if present
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Normalize pixel values
    image_array = image_array.astype(
        np.float32
    ) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Find class with highest probability
    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    # Store probability for every class
    probabilities = {}

    for i, class_name in enumerate(CLASS_NAMES):
        probabilities[class_name] = float(
            predictions[i]
        )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }