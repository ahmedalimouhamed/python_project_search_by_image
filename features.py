import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D

import numpy as np
from numpy.linalg import norm

import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable=False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])
model.summary()

# img = cv2.imread("1636.jpg")
# img = cv2.resize(img, (224, 224))
# img = np.array(img)
# expand_img = np.expand_dims(img, axis=0)
# pre_img = preprocess_input(expand_img)
# result = model.predict(pre_img).flatten()
# normalized = result/norm(result)

def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

# print(extract_feature("1636.jpg", model))

filename = []
feature_list = []

for file in os.listdir('dataset'):
    filename.append(os.path.join('dataset', file))

for file in tqdm(filename):
    feature_list.append(extract_feature(file, model))

pickle.dump(feature_list, open('featurevector.pkl', 'wb'))
pickle.dump(filename, open('filenames.pkl', 'wb'))




