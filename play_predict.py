"""


The script loads the trained model and predicts the hand sign in the image. It performs the following:
 a)	The tested image is to be supplied via the arguments list 
b)	visualisation of the supplied image with the prediction score and predicted label. This is also saved in the folder where the program is located

To use this script via the argument list(CMD terminal), enter the following:

python play_predict.py image.png best_model.hdf5

play_predict.py is the name of this program file. image.png is the image file fed in for prediction 
best_model.hdf5 is the best saved model generated from part 1 of the coursework and used to predict the images.

This script was developed by Zuby Madu with help from Deep Learning for Vision Systems by Mohamed Elgendy, 
Github resources such as https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb,
https://github.com/KeithGalli/neural-nets/blob/master/real_world_example.ipynb, Stackoverflow, https://machinelearningmastery.com/keras-functional-api-deep-learning/
https://www.malicksarr.com/split-train-test-validation-python/, keras.io and Google.com    
    
    
"""


from  argparse import ArgumentParser
from keras import models
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the command line arguments
parser = ArgumentParser(description='Hand Sign Prediction')
parser.add_argument('image_path', metavar='path', type=str, help='Path to the image file')
parser.add_argument('model_path', metavar='path', type=str, help='Path to the trained model file')
args = parser.parse_args()

# Load the model
model = models.load_model(args.model_path)

# Load and preprocess the image
image = Image.open(args.image_path).convert('RGB')
image = image.resize((224, 224))
image = np.array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)


# Predict the class probabilities
preds = model.predict(image)[0]
class_index = np.argmax(preds)
class_label = ['Rock', 'Paper', 'Scissors'][class_index]
score = preds[class_index]

# Display the image and predicted class label with score
plt.imshow(Image.open(args.image_path))
plt.title(f'Predicted class: {class_label} with score: {score}')
plt.axis('on')
plt.savefig('prediction.png')
plt.show()
