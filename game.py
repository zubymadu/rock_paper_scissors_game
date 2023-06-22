
"""


The script is required to perform the following:
a)	Read two images of hand signs provided as script arguments 
b)	Predict the labels of the two images 
c)	Output which image won the rock, paper, scissor game 

To use this script via the argument list(CMD terminal), enter the following:

python play_game.py image2.png image1.png best_model.hdf5

play_game.py is the name of this program file
image2.png and image1.png are the image files fed in for the game 
best_model.hdf5 is the best saved model generated from part 1 of the coursework and used to predict the images.

This script was developed by Zuby Madu with help from Deep Learning for Vision Systems by Mohamed Elgendy, 
Github resources such as https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb,
https://github.com/KeithGalli/neural-nets/blob/master/real_world_example.ipynb, Stackoverflow, https://machinelearningmastery.com/keras-functional-api-deep-learning/
https://www.malicksarr.com/split-train-test-validation-python/, keras.io and Google.com    
    
    
"""

from argparse import ArgumentParser
from keras import models
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Define the command line arguments
parser = ArgumentParser(description='Rock-Paper-Scissors')
parser.add_argument('image1_path', metavar='path', type=str, help='Path to the first image file')
parser.add_argument('image2_path', metavar='path', type=str, help='Path to the second image file')
parser.add_argument('model_path', metavar='path', type=str, help='Path to the trained model file')
args = parser.parse_args()

# Load the model
model = models.load_model(args.model_path)

# Load and preprocess the images
images = []
for image_path in [args.image1_path, args.image2_path]:
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    img = np.squeeze(img)
    images.append(img)

# Predict the class labels
preds = model.predict(np.array(images))
class_indices = np.argmax(preds, axis=1)
class_labels = ['Paper', 'Rock', 'Scissors']

# Convert the class indices to class labels
class_labels = [class_labels[i] for i in class_indices]

# Determine the winner
if class_labels[0] == class_labels[1]:
    result = "It's a tie!"
    winner_label = ""
    winning_image_idx = None
else:
    if class_labels[0] == 'Rock':
        if class_labels[1] == 'Paper':
            result = "Image 2 wins! Paper beats Rock"
            winner_label = class_labels[1]
            winning_image_idx = 1
        else:
            result = "Image 1 wins! Rock beats Scissors"
            winner_label = class_labels[0]
            winning_image_idx = 0
    elif class_labels[0] == 'Paper':
        if class_labels[1] == 'Scissors':
            result = "Image 2 wins! Scissors beat Paper"
            winner_label = class_labels[1]
            winning_image_idx = 1
        else:
            result = "Image 1 wins! Paper beats Rock"
            winner_label = class_labels[0]
            winning_image_idx = 0
    else:
        if class_labels[1] == 'Rock':
            result = "Image 2 wins! Rock beats Scissors"
            winner_label = class_labels[1]
            winning_image_idx = 1
        else:
            result = "Image 1 wins! Scissors beat Paper"
            winner_label = class_labels[0]
            winning_image_idx = 0

# Display the predicted labels and winner
print(f"Image 1: {class_labels[0]}")
print(f"Image 2: {class_labels[1]}")
print(result)


# Display the predicted labels and result on the images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, image_path in enumerate([args.image1_path, args.image2_path]):
    img = Image.open(image_path)
    axs[i].imshow(img)
    axs[i].axis('on')
    if winner_label == "":
        axs[i].set_title(class_labels[i])
    else:
        if class_labels[i] == winner_label:
            axs[i].set_title(f"{class_labels[i]} (winner)")
        else:
            axs[i].set_title(class_labels[i])
        
fig.suptitle(result)
plt.show()