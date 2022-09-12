"""Imports"""
#Ignore no GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Import necessary libraries
import tensorflow as tf
import json
import argparse
from PIL import Image
import numpy as np
import tensorflow_hub as hub

"""Argparse"""
parser = argparse.ArgumentParser(description='This is a flower image classifier')
#Adding arguments
parser.add_argument('filepath', help='Input Image file path')
parser.add_argument('model',  help='Input Model file path')
parser.add_argument('--top_k', type=int, help='Input top matches number')
parser.add_argument('--category_names', help='JSON Category names file path')
args = parser.parse_args()

print('\nPredicting........\n')
"""Functions"""
#Pre-processing image before prediction
def process_image(image):
    image_size=224
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

#Prediction function
def predict(filepath, model, top_k):
    im = Image.open(filepath)
    im = np.asarray(im)
    im_process = process_image(im)
    im_expand = np.expand_dims(im_process, axis=0)
    
    pred = model.predict(im_expand)
    top_k_probs, top_k_classes = tf.nn.top_k(pred, k=top_k)
    
    top_k_probs = top_k_probs.numpy()
    top_k_classes = top_k_classes.numpy()
    
    return top_k_probs, top_k_classes

def flower_names(cat_file, class_indices):
    with open(cat_file, 'r') as f:
        class_names = json.load(f)
    flowernames = [class_names[str(i+1)] for i in class_indices[0]]

    return flowernames

"""Image and model loading"""
image_path = args.filepath
load_Model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})

"""printing the results"""
#If top_k option not use, top_k value default to 1; if category names is used

if not args.top_k:
    args.top_k = 1
    probs, class_indices = (predict(image_path, load_Model, args.top_k))
    print('\nThe probabilities are: ', probs)
    print('\nThe classes are: ', class_indices, '\n')
    if args.category_names:
        print('The flower names are: ', flower_names(args.category_names, class_indices), '\n')
    else:
        None
else:
    probs, class_indices = (predict(image_path, load_Model, args.top_k))
    print('\nThe probabilities are: ', probs)
    print('\nThe classes are: ', class_indices, '\n')
    if args.category_names:
        print('The flower names are: ', flower_names(args.category_names, class_indices), '\n')
    else:
        None       
 