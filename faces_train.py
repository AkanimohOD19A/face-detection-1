import os
import cv2
import pickle
import numpy as np
from PIL import Image

# Create Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

##-> Training Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cascade
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# Containers, to get labels and img-dir respectively
current_id = 0
label_ids = {}
y_labels = []
x_train = []

# Loop through directory
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith(".jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            pil_image = Image.open(path).convert("L") # Library - converts to grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS) # Resize Images
            image_array = np.array(pil_image, "uint8") # Convert Image into numpy array
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

## Pickle Label IDs
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

## Train a model
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
