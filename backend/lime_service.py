import numpy as np
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries

from model_service import model


explainer = lime_image.LimeImageExplainer()


def predict_for_lime(images):
    images = np.array(images).astype(np.float32) / 255.0
    return model.predict(images, verbose=0)


def explain_image(image: Image.Image):

    image = image.resize((224, 224))
    image_array = np.array(image.convert("RGB"))

    explanation = explainer.explain_instance(
        image_array,
        predict_for_lime,
        top_labels=1,
        hide_color=0,
        num_samples=100
    )

    predicted_index = np.argmax(
        model.predict(
            np.expand_dims(
                image_array.astype(np.float32) / 255.0,
                axis=0
            ),
            verbose=0
        )[0]
    )

    temp, mask = explanation.get_image_and_mask(
        predicted_index,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    lime_image_array = mark_boundaries(
        temp,
        mask
    )

    lime_image_array = (
        lime_image_array * 255
    ).astype(np.uint8)

    return lime_image_array