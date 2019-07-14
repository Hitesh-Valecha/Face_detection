import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []  #numbers related to labels
x_train = []  #numbers of actual pixel values

for root, dirs,  files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            # os.path.dirname(path) = root {you can use root instead}
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            #print(label_ids)
            #y_labels.append(label) # some number value for labels
            #x_train.append(path) #verify this image, turn into a NUMPY array, turn GRAY
            pil_image = Image.open(path).convert("L") # L converts image to grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8") 
            #print(image_array)
            """every image has pixel values,
                first we turn images into grayscale then we turned that grayscale image into a numpy array
                and we use that as a list of numbers that are related to this image
                and with that then we can actually start training it"""
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

#pickle files can be saved with any extension pkg,picle not pickle as compulsory
with open("labels.pickle", 'wb') as f:  #f is file & wb is writing bytes
    pickle.dump(label_ids, f)           #dump label_ids to that file

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")