import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:  #f is file & rb is reading bytes
    orginal_labels = pickle.load(f)
    labels = {v:k for k,v in orginal_labels.items()}    #k is key & v is value


cap = cv2.VideoCapture(0)

while(True):
    #capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]   #ROI is region of interest
        roi_color = frame[y:y+h, x:x+w]

        # recognize? deep learned model predict keras or tensorflow or pytorch or scikit-learn
        id_, conf = recognizer.predict(roi_gray)    #id_ is givig label back & conf is confidence
        if conf>=45 and conf<=85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255) #white
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "1.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) #BGR (0-255)
        stroke = 2 #thickness of line
        end_cord_x = x + w  #width
        end_cord_y = y + h  #height
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,255,0), 2)
        
    #Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()