from collections import defaultdict
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
import os
from sklearn.model_selection import train_test_split
import click
import kagglehub
import constants


def download_dataset() -> str:
    """
    Download the dataset from Kaggle
    """
    path = kagglehub.dataset_download("anshtanwar/pets-facial-expression-dataset")
    click.echo(f"Path to dataset files: {path}")
    return path


def build_labeled_images(path: str) -> dict[str, list[np.ndarray]]:
    """
    Build labels for the dataset
    """
    image_dirs = {
        "happy": f"{path}/happy",
        "sad": f"{path}/Sad",
        "angry": f"{path}/Angry"
    }

    # Key is the label, value is a list of images
    labeled_images = defaultdict(list)

    for label, folder in image_dirs.items():
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                labeled_images[label].append(img)

    found_images = ''.join(f"{label}: {len(images)}\n" for label, images in labeled_images.items())
    click.echo(f"Found images:\n{found_images}")

    # Order matters, see constants.LABEL_TO_INDEX
    return labeled_images


def resize(labeled_images: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize the images to the input shape
    Returns:
        data: np.ndarray - Array of resized images
        labels: np.ndarray - Array of labels
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

    data = []
    labels = []

    found_face, not_found_face = 0, 0

    for label, image_list in labeled_images.items():
        for image in image_list:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cropped_face = image[y:y + h, x:x + w]
                resized_face = cv2.resize(cropped_face, (128, 128))
                data.append(resized_face)
                labels.append(constants.LABEL_TO_INDEX[label])
                found_face += 1
            else:
                not_found_face += 1
                resized_face = cv2.resize(image, (128, 128))
                data.append(resized_face)
                labels.append(constants.LABEL_TO_INDEX[label])

    print(f"Found faces: {found_face}, Not found faces: {not_found_face}")

    data = np.array(data) / 255.0

    labels = np.array(labels)
    return data, labels


def build_generators(X_train, X_val, y_train, y_val, data, labels) -> tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Build generators for the dataset
    Returns:
        train_generator: ImageDataGenerator - Generator for the training set
        val_generator: ImageDataGenerator - Generator for the validation set
    """
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_datagen.fit(X_train)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_datagen = ImageDataGenerator(rotation_range=40,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest'
                                     )
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # test_datagen = ImageDataGenerator(rotation_range=40,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest'
    # )
    # test_generator = test_datagen.flow(data, labels, batch_size=32)

    return train_generator, val_generator


def build_model(train_generator, val_generator) -> tuple[tf.keras.Model, dict]:
    """
    Build the model
    Returns:
        model: tf.keras.Model - The model
    """
    ResNet50V2 = tf.keras.applications.ResNet50V2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )

    model = Sequential([
        ResNet50V2,
        GlobalAveragePooling2D(),
        Dropout(0.25),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.00015), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=[early_stopping, lr_reduction]
    )

    return (model, history)


def print_test_data_evaluation(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """
    Print the evaluation of the model on the test data
    """
    loss, accuracy = model.evaluate(X_val, y_val)
    losstr, accuracytr = model.evaluate(X_train, y_train)
    click.echo(f"Test data evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")
    click.echo(f"Train data evaluation: loss={losstr:.4f}, accuracy={accuracytr:.4f}")


@click.command()
@click.option("--model_filename", default="model.keras", help="Path to model name to save")
def init(model_filename: str):
    """
    Build the model
    Returns:
        model: tf.keras.Model - The model
    """
    path = download_dataset()

    labels = build_labeled_images(path)
    data, labels = resize(labels)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    train_generator, val_generator = build_generators(X_train, X_val, y_train, y_val, data, labels)

    model, _ = build_model(train_generator, val_generator)
    print_test_data_evaluation(model, X_train, y_train, X_val, y_val)

    model.save(f'./{model_filename}')
    click.echo(f"Model saved to {model_filename}")


if __name__ == '__main__':
    init()
