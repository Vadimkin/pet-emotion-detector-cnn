import tensorflow as tf
import click

@click.command()
@click.option("--model_filename", default="model.keras", help="Model filename to use")
def print_model_summary(model_filename: str):
    model = tf.keras.models.load_model(model_filename)
    model.summary()

    

if __name__ == '__main__':
    print_model_summary()