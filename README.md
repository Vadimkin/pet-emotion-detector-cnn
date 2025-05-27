# "Pet's Facial Expression Poem" Bot Generator

Simple telegram bot that can generate a poem about your pet's facial expression using a trained ResNet model.
The model is trained on the [Pets Facial Expression Dataset](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset) from Kaggle. Additionaly, I'm using cv2 to detect the face and resize it to the input shape of the model to increase the accuracy of the model.

## Quick start
```bash
# Tensorflow is not supported by python3.13 yet
python3.12 -m venv venv
. venv/bin/activate

pip install -r requirements.txt

# There is notebook file with the model summary and the data visualization.
./venv/bin/jupyter notebook
# ... and then open Visualization.ipynb file
# ... or you can build the model via cli tool
python build_model.py
# You should see created "model.keras" file

cp .env.example .env
# Put your telegram and openai keys in .env file
nano .env

# Run the bot
python run.py
```
