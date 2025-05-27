import cv2
import numpy as np
import tensorflow as tf

import constants


def predict_pet_expression(image_path: str) -> tuple[str | None, float | None]:
    label_map = {v: k for k, v in constants.LABEL_TO_INDEX.items()}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    model = tf.keras.models.load_model(constants.MODEL_FILENAME)

    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cropped_face = image[y:y + h, x:x + w]
        resized_face = cv2.resize(cropped_face, (128, 128))
        image = resized_face
    else:
        image = cv2.resize(image, (128, 128))

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    # prediction is a 1x3 array, we need to get the index of the highest probability
    predicted_class = np.argmax(prediction)

    # Return only if there is more than 50% confidence
    if prediction[0][predicted_class] < 0.5:
        print("Prediction confidence is too low.")
        return None, None

    probability = float(prediction[0][predicted_class])
    print(f"Predicted class: {label_map[predicted_class]} (probability: {probability:.2f})")
    return label_map[predicted_class], probability
