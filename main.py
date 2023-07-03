import os
import cntk as C
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input

# Set up the paths and filenames
model_path = 'path_to_your_model'
vocab_path = 'path_to_your_vocabulary'

# Load the vocabulary
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = f.read().splitlines()

# Load the trained model
model = C.load_model(model_path)

# Set the model's input variable
input_var = model.arguments[0]

# Define the preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust the image size as per your model's requirements
    img = np.array(img)
    img = preprocess_input(img)
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # Transpose the image array
    return img

# Generate caption for an image
def generate_caption(image_path):
    img = preprocess_image(image_path)
    caption = []
    while True:
        word_idx = model.eval({input_var: [img]})[0].argmax()
        word = vocab[word_idx]
        caption.append(word)
        if word == '<end>':
            break
        img = np.concatenate((img[1:], np.expand_dims(np.array(word_idx, dtype=np.float32), axis=0)))
    return ' '.join(caption[1:-1])  # Exclude <start> and <end> tokens

# Test the image caption generator
image_path = 'path_to_your_image'
caption = generate_caption(image_path)
print("Caption:", caption)
