import click
import cv2
import numpy as np
import tensorflow as tf

import constants

@click.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--model_filename", default="model.keras", help="Model filename to use")
def predict_pet_expression(image: str, model_filename: str):
    label_map = {v: k for k, v in constants.LABEL_TO_INDEX.items()}
    model = tf.keras.models.load_model(model_filename)

    image = cv2.imread(image)
    image = cv2.resize(image, constants.INPUT_SHAPE[:2])
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    # prediction is a 1x3 array, we need to get the index of the highest probability
    predicted_class = np.argmax(prediction)
    click.echo(f"Predicted class: {label_map[predicted_class]} (probability: {prediction[0][predicted_class]:.2f})")

if __name__ == '__main__':
    predict_pet_expression()